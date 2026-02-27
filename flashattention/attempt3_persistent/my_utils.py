import math
from typing import Callable, Type, Union, Optional
import cutlass
from cutlass import cute, const_expr, Int32, Boolean
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cutlass_dsl import Numeric, dsl_user_op, T
from cutlass.utils import LayoutEnum
from cutlass._mlir.dialects import nvvm, llvm

def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))

def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))

def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx



def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
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


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))

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


@dsl_user_op
def fmax(a: float | cutlass.Float32, b: float | cutlass.Float32, c: float | cutlass.Float32 | None = None, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmax(
            T.f32(),
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            c=cutlass.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )

@cute.jit
def fmax_reduce(x: cute.TensorSSA, init_val: float | cutlass.Float32 | None = None) -> cutlass.Float32:
    res= cute.make_rmem_tensor(x.shape, cutlass.Float32)
    res.store(x)

    # allocate 4 registers, do 4 maxes at a time and then tree-reduce at the end
    # not sure why they chose a factor of 4, might just be empirically the best
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4): # start stop step
        local_max[0] = fmax(local_max[0], res[i+0])
        local_max[1] = fmax(local_max[1], res[i+1])
        local_max[2] = fmax(local_max[2], res[i+2])
        local_max[3] = fmax(local_max[3], res[i+3])
    local_max[0] = fmax(local_max[0], local_max[1])
    local_max[2] = fmax(local_max[2], local_max[3])
    local_max[0] = fmax(local_max[0], local_max[2])
    return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)

@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric, # SSA : static single assignment(?)
    op: Callable, 
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        # this is if you're trying to reduce a whole matrix, we just loop through each element individually and return the result
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        # for a number, we just butterfly reduce this
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def exp2f(x: cute.TensorSSA | cutlass.Float32) -> cute.TensorSSA | cutlass.Float32:
    """exp2f calculation for both vector and scalar.
    :param x: input value
    :type x: cute.TensorSSA or Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or Float32
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_rmem_tensor(x.shape, cutlass.Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.math.exp2(res[i], fastmath=True)
        return res.load()
    else:
        return cute.math.exp2(x, fastmath=True)

@cute.jit
def fadd_reduce(
    x: cute.TensorSSA, init_val: float | cutlass.Float32 | None = None
) -> cutlass.Float32:
    # sum reduction
    if const_expr(init_val is None):
        init_val = cutlass.Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


@dsl_user_op
def cvt_f16x2_f32(
    a: float | cutlass.Float32, b: float | cutlass.Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [cutlass.BFloat16, cutlass.Float16], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
    """Convert Float32 tensor to Float16/BFloat16.

    Args:
        src: Source tensor with Float32 element type
        dst_or_dtype: Either a destination tensor or a dtype (Float16/BFloat16)

    Returns:
        None if dst is a tensor, or a new tensor if dtype is provided
    """
    if const_expr(isinstance(dst_or_dtype, type)):
        # dtype variant: create new tensor and call the tensor variant
        dtype = dst_or_dtype
        dst = cute.make_fragment(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        # tensor variant: write to dst
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(src.shape), "dst and src must have the same size"
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [cutlass.BFloat16, cutlass.Float16], (
            "dst must be BFloat16 or Float16"
        )
        assert src.element_type is cutlass.Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)