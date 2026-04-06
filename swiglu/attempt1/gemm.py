import cutlass
from cutlass import cute, pipeline
from cck.runtime import *
import torch

class Kernel:
  def __init__(self, ):
    self.nwarps = 12

  @cute.jit
  def __call__(self, W: cute.Tensor, V: cute.Tensor, X: cute.Tensor, O: cute.Tensor, stream: cuda.CUstream):
    W = layout.select_and_combine_batch_dim(W, (1, 2, 0))
    V = layout.select_and_combine_batch_dim(V, (1, 2, 0))
    X = layout.select_and_combine_batch_dim(X, (1, 2, 0))
    O = layout.select_and_combine_batch_dim(O, (1, 2, 0))
    Ws_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    Vs_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)
    Xs_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, 2)

    scheduler_params = tile_scheduler.Gemm2DTileScheduler.to_underlying_arguments(
      tile_scheduler.Gemm2DTileSchedulerArguments.create(O, 128, 128, True))
    scheduler_grid = tile_scheduler.Gemm2DTileScheduler.get_grid_shape(scheduler_params, 132)
    gemmw_tiled_gemm = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128)
    gemmv_tiled_gemm = mma.get_tiled_mma(cutlass.BFloat16, True, True, cutlass.Float32, 128, 128)
    X_g2s_tma_atom, X_g2s_tma_tensor = shared.get_tma_tensor_and_atom(X, Xs_layout, 128, 64)
    V_g2s_tma_atom, V_g2s_tma_tensor = shared.get_tma_tensor_and_atom(V, Vs_layout, 128, 64)
    W_g2s_tma_atom, W_g2s_tma_tensor = shared.get_tma_tensor_and_atom(W, Ws_layout, 128, 64)
    self.kernel(V, V_g2s_tma_atom, V_g2s_tma_tensor, Vs_layout, W, W_g2s_tma_atom, W_g2s_tma_tensor, Ws_layout, X, X_g2s_tma_atom, X_g2s_tma_tensor, Xs_layout, gemmv_tiled_gemm, gemmw_tiled_gemm, scheduler_params).launch(grid=scheduler_grid, block=[self.nwarps * 32], stream=stream) # no cluster for now

  @cute.kernel
  def kernel(self, V, V_g2s_tma_atom, V_g2s_tma_tensor, Vs_layout, W, W_g2s_tma_atom, W_g2s_tma_tensor, Ws_layout, X, X_g2s_tma_atom, X_g2s_tma_tensor, Xs_layout, gemmv_tiled_gemm, gemmw_tiled_gemm, scheduler_params): # self.nwarps warps
    SharedStorage = type("SharedStorage", (), dict())
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    normalized_warp_idx = warp_idx
    tidx, _, _ = cute.arch.thread_idx()
    SharedStorage.__annotations__['Ws_ptr'] = cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(Ws_layout)], 1024]
    SharedStorage.__annotations__['Vs_ptr'] = cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(Vs_layout)], 1024]
    SharedStorage.__annotations__['Xs_ptr'] = cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(Xs_layout)], 1024]
    SharedStorage.__annotations__['pipew_ptr'] = cute.struct.MemRange[cutlass.Int64, 2 * 2]
    SharedStorage.__annotations__['pipev_ptr'] = cute.struct.MemRange[cutlass.Int64, 2 * 2]
    SharedStorage.__annotations__['pipex_ptr'] = cute.struct.MemRange[cutlass.Int64, 2 * 2]
    smem_allocator_ = cutlass.utils.SmemAllocator()
    smem__ = smem_allocator_.allocate(cute.struct(SharedStorage))
    Ws = smem__.Ws_ptr.get_tensor(Ws_layout.outer, swizzle=Ws_layout.inner)
    Vs = smem__.Vs_ptr.get_tensor(Vs_layout.outer, swizzle=Vs_layout.inner)
    Xs = smem__.Xs_ptr.get_tensor(Xs_layout.outer, swizzle=Xs_layout.inner)
    pipew_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
    pipew_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=0)
    pipew = pipeline.PipelineTmaAsync.create(
      barrier_storage=smem__.pipew_ptr.data_ptr(),
      num_stages=2,
      producer_group=pipew_prod_grp,
      consumer_group=pipew_cons_grp,
      tx_count=cute.size_in_bytes(cutlass.BFloat16, cute.select(Ws_layout, mode=[0, 1])),
      defer_sync=False,
    )
    pipev_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
    pipev_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=0)
    pipev = pipeline.PipelineTmaAsync.create(
      barrier_storage=smem__.pipev_ptr.data_ptr(),
      num_stages=2,
      producer_group=pipev_prod_grp,
      consumer_group=pipev_cons_grp,
      tx_count=cute.size_in_bytes(cutlass.BFloat16, cute.select(Vs_layout, mode=[0, 1])),
      defer_sync=False,
    )
    pipex_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
    pipex_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=0)
    pipex = pipeline.PipelineTmaAsync.create(
      barrier_storage=smem__.pipex_ptr.data_ptr(),
      num_stages=2,
      producer_group=pipex_prod_grp,
      consumer_group=pipex_cons_grp,
      tx_count=cute.size_in_bytes(cutlass.BFloat16, cute.select(Vs_layout, mode=[0, 1])),
      defer_sync=False,
    )
    scheduler = Gemm2DTileScheduler.create(scheduler_params)
    if (warp_idx < 8): # [None, 8) --> (8 warps)
      # normalized_warp_idx unchanged
      # tidx unchanged
      cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
      gemmw = mma.get_acc(gemmw_tiled_gemm, 128, 128, cutlass.Float32)
      gemmw_should_accumulate = False
      gemmv = mma.get_acc(gemmv_tiled_gemm, 128, 128, cutlass.Float32)
      gemmv_should_accumulate = False
      w1_work_tile = scheduler.initial_work_tile_info()
      while w1_work_tile.is_valid_tile:
        w1_tile_coord = w1_work_tile.tile_idx
        for k in cutlass.range(4, unroll=1):
          pipex.consumer_wait(cstate)
          pipev.consumer_wait(cstate)
          assert cutlass.const_expr(8 == ((128 // 64) * 4)), f'Gemm expected {((128 // 64) * 4)} warps, got {8}'
          mma.accumulating_gemm_ss(tidx, gemmv_tiled_gemm, Xs, Vs, gemmv, cstate, cstate, gemmv_should_accumulate)
          gemmv_should_accumulate = True
          pipew.consumer_wait(cstate)
          assert cutlass.const_expr(8 == ((128 // 64) * 4)), f'Gemm expected {((128 // 64) * 4)} warps, got {8}'
          mma.accumulating_gemm_ss(tidx, gemmw_tiled_gemm, Xs, Ws, gemmw, cstate, cstate, gemmw_should_accumulate)
          gemmw_should_accumulate = True
        # sigmoid + whatever epilogue here
        scheduler.fetch_next_work()
        scheduler.advance_to_next_work()
        w1_work_tile = scheduler.get_current_work()
    if (8 <= warp_idx < (8 + 4)): # [8, (8 + 4)) --> (4 warps)
      normalized_warp_idx = warp_idx - 8
      tidx = tidx - (8 * cute.arch.WARP_SIZE)
      pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
      w2_work_tile = scheduler.initial_work_tile_info()
      while w2_work_tile.is_valid_tile:
        w2_tile_coord = w2_work_tile.tile_idx
        XSlice = X[None, None, w2_tile_coord[3]]
        VSlice = V[None, None, w2_tile_coord[3]]
        WSlice = W[None, None, w2_tile_coord[3]]
        for k1 in cutlass.range(4, unroll=1):
          X_g2s_tma_tensor_slice = X_g2s_tma_tensor[None, None, w2_tile_coord[3]]
          if warp_idx == 0: # [None, 1)
            # normalized_warp_idx unchanged
            # tidx unchanged
            pipex.producer_acquire(pstate)
            shared.tma_copy(X_g2s_tma_atom, X_g2s_tma_tensor, Xs, 128, 64, w2_tile_coord[0], k1, pipex, pstate)
          V_g2s_tma_tensor_slice = V_g2s_tma_tensor[None, None, w2_tile_coord[3]]
          if warp_idx == 0: # [None, 1)
            # normalized_warp_idx unchanged
            # tidx unchanged
            pipev.producer_acquire(pstate)
            shared.tma_copy(V_g2s_tma_atom, V_g2s_tma_tensor, Vs, 128, 64, w2_tile_coord[1], k1, pipev, pstate)
          W_g2s_tma_tensor_slice = W_g2s_tma_tensor[None, None, w2_tile_coord[3]]
          if warp_idx == 0: # [None, 1)
            # normalized_warp_idx unchanged
            # tidx unchanged
            pipew.producer_acquire(pstate)
            shared.tma_copy(W_g2s_tma_atom, W_g2s_tma_tensor, Ws, 128, 64, w2_tile_coord[1], k1, pipew, pstate)
        scheduler.fetch_next_work()
        scheduler.advance_to_next_work()
        w2_work_tile = scheduler.get_current_work()