import math
import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32
import utils


# This only accepts a numeric
# cute.arch.warp_reduction

@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable, 
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE
) -> cute.TensorSSA | cute.Numeric:
    """
    Reduces a value or matrix of values across the entire warp

    All threads end up with the reduced value
    """
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        # reduce the whole matrix elementwise
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        # for a number, we just butterfly reduce this
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    """
    Reduction buffer is (num_warps / warps_per_row, warps_per_row)
    Each warp will reduce from the last dimension.

    Every 0th thread outputs to SMEM, and then threads grab and reduce values
    """
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier() # TODO swap this to not use syncthreads in case we use warp spec.

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def cluster_reduce(
        val: cute.Numeric, op: Callable, 
        reduction_buffer: cute.Tensor, 
        mbar_ptr: cute.Pointer, init_val: cute.Numeric=0.0,
        phase: Optional[cutlass.Int32] = None):
    """
    Reduction buffer is (num_warps / warps_per_row, (warps_per_row, cluster_n))

    You first store all warps' data to cluster memory then each cluster reduces
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if (warp_idx == 0):
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        # cluster_n threads store to each other cluster CTA, for each warp
        utils.store_shared_remote(
            val,
            utils.elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE) # number of iters to capture all the values
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return cute.arch.warp_reduction(block_reduce_val, op)

@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[cutlass.Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if cutlass.const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(val, op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val)


# TODO if I want to target hopper I will need cluster reduce
@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[cutlass.Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps/warps_per_row, (warps_per_row, cluster_n))"""
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        # Assume that at a thread level you want to reduce x
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if cutlass.const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = cute.arch.warp_reduction(val, warp_op, threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE))
    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, "must provide mbar ptr for cluster reduce"
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(val, warp_op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val)
    return val