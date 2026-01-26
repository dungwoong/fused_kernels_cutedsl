from typing import Type, Union, Optional
import cutlass
from cutlass import cute, const_expr, Int32, Boolean
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.utils import LayoutEnum

def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))

def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))

def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool=False,
    **kwargs,
):
    """Returns a callable to perform the G2S copy"""
    src_is_smem = cutlass.const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)

    s, g = cute.nvgpu.cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - (1 if not single_stage else 0)),
        cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - (1 if not single_stage else 0)),
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **kwargs2):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **kwargs2, **kwargs)
    
    def copy_tma_single_stage(**kwargs2):
        cute.copy(atom, src, dst, **kwargs, **kwargs2)
    return (copy_tma if not single_stage else copy_tma_single_stage), s, g


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    stage: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout == LayoutEnum.COL_MAJOR) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged

# I ONLY use this for epi but in original quack codebase they use this for AB smem layout too(maybe)
make_smem_layout_epi = make_smem_layout

@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
) -> None:
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])): # m, k, n_iters
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    cute.nvgpu.warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        cute.nvgpu.warpgroup.wait_group(wg_wait)

@cute.jit
def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int=-1,
) -> cute.Tensor:
    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), cutlass.Float32)
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
    return acc

@cute.jit
def gemm_w_index(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int=-1,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)

def get_smem_store_atom(
    arch: cutlass.Constexpr[int], element_type: Type[cute.Numeric], transpose: bool = False
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )

@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    l = cute.logical_divide(
        acc_layout, ((None, None, 2), None, None)
    )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
    rA_mma_view = cute.make_layout(
        (
            (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
            l.shape[1],
            (l.shape[0][2][1], l.shape[2]),
        ),
        stride=(
            (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
            l.stride[1],
            (l.stride[0][2][1], l.stride[2]),
        ),
    )
    return rA_mma_view