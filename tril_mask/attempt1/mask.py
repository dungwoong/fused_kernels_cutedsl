# assume layout is ((2, MMA_M), (2, V, MMA_N))
# e.g. 128x64 is (2, 2) for the 16x16 tile, then V=4, MMA_M=MMA_N=1

# you can just compute a col limit
import cutlass
from cutlass import const_expr, cute
from typing import Type


def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
    quack layout_utils.py
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)

def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))

@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    """
    You sync <value> with <width> threads
    """
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # important: need stride 1 and not 0 for recast_tensor to work
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@cute.jit
def causal_mask(thr_mma: cute.TiledMma, gemm_acc: cute.Tensor, tile_m: cutlass.Int32, tile_n: cutlass.Int32, idx_m: cutlass.Int32, idx_n: cutlass.Int32, val: Type[cutlass.Numeric]):
    """
    Assume that this is going to be row-major
    """
    gemm_acc_mn = reshape_acc_to_mn(gemm_acc)
    acc_shape = (tile_m, tile_n)
    coords_ = cute.make_identity_tensor(acc_shape)

    # you can only do this because their cute ir has partitioning for MMA
    coords = reshape_acc_to_mn(thr_mma.partition_C(coords_))

    print('gemm_acc:',gemm_acc)
    print('gemm_acc_mn:', gemm_acc_mn)
    """
    Logic:
    - start row is idx_m * tile_n
    - start col is idx_n * tile_n
    - row_idx adds coord to start row
    - max col idx, just subtract start col
    - pray CSE gets this
    """

    for r in cutlass.range(cute.size(coords.shape[0]), unroll_full=True):
        row_idx = coords[r, 0][0] + idx_m * tile_m
        col_limit_right = row_idx - idx_n * tile_n
        for c in cutlass.range(cute.size(coords.shape[1]), unroll_full=True):
            gemm_acc_mn[r, c] = (
                val
                if coords[0, c][1] > col_limit_right
                else gemm_acc_mn[r, c]
            )
