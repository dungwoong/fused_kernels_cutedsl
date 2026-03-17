import cutlass
from cutlass import cute
from cutlass import Boolean, const_expr, Int32
from cutlass.cute.nvgpu import warpgroup
from typing import Optional, Type
import cutlass.utils.hopper_helpers as sm90_utils


def get_tiled_mma(ab_dtype: Type[cutlass.Numeric], a_major_k, b_major_k, acc_dtype, tile_m, tile_n):
    assert tile_m % 64 == 0, "tiled_mma tile_m must be a multiple of 64 for now"
    assert tile_n % 8 == 0, "tiled_mma tile_n must be a multiple of 8 for now"
    # if tile shape is none then you just do the entire tile size
    a_mode = cute.nvgpu.warpgroup.OperandMajorMode.K if a_major_k else cute.nvgpu.warpgroup.OperandMajorMode.MN
    b_mode = cute.nvgpu.warpgroup.OperandMajorMode.K if b_major_k else cute.nvgpu.warpgroup.OperandMajorMode.MN
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        ab_dtype,
        ab_dtype,
        a_mode,
        b_mode,
        acc_dtype,
        # TODO this assumes the MMA atom layout
        atom_layout_mnk=(tile_m // 64, 1, 1),
        tiler_mn=(64, tile_n),
    )
    return tiled_mma


def get_acc(tiled_mma: cute.TiledMma, tile_m: int, tile_n: int, dtype: Type[cutlass.Numeric]):
    thr_mma = tiled_mma.get_slice(0)
    acc_shape = thr_mma.partition_shape_C((tile_m, tile_n))
    acc = cute.make_rmem_tensor(acc_shape, dtype)
    return acc


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
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):  # m, k, n_iters
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
    wg_wait: int = -1,
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
    wg_wait: int = -1,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


@cute.jit
def accumulating_gemm_ss(
    tidx: int,
    tiled_mma: cute.TiledMma,
    sA: cute.Tensor,
    sB: cute.Tensor,
    acc: cute.Tensor,
    a_state: cutlass.pipeline.PipelineState,
    b_state: cutlass.pipeline.PipelineState,
    accumulate: bool,
    wg_wait: int = 0,
):
    thr_mma = tiled_mma.get_slice(tidx)
    tSrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tSrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    gemm_w_index(
        tiled_mma,
        acc,
        tSrA,
        tSrB,
        not accumulate,
        A_idx=a_state.index,
        B_idx=b_state.index,
        wg_wait=wg_wait,
    )
