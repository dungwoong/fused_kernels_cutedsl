import cutlass
from cutlass import cute, pipeline
from cck.runtime import *
import torch
import cuda.bindings.driver as cuda

# CURRENTLY REFACTORING STAGES

class Kernel:
  def __init__(self, ):
    """
    Always going to be bfloat16, fp32 accum
    """
    self.nwarps = 12
    self.stages = 2
    self.num_consumer_warps = 8
    self.num_producer_warps = 4

  @cute.jit
  def __call__(self, W: cute.Tensor, V: cute.Tensor, X: cute.Tensor, O: cute.Tensor, stream: cuda.CUstream):
    W = layout.select_and_combine_batch_dim(W, (1, 2, 0))
    V = layout.select_and_combine_batch_dim(V, (1, 2, 0))
    X = layout.select_and_combine_batch_dim(X, (1, 2, 0))
    O = layout.select_and_combine_batch_dim(O, (1, 2, 0))
    Ws_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, self.stages)
    Vs_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, self.stages)
    Xs_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, 128, 64, self.stages)

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
    
    # Make SMEM
    SharedStorage = type("SharedStorage", (), dict())
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    normalized_warp_idx = warp_idx
    tidx, _, _ = cute.arch.thread_idx()
    SharedStorage.__annotations__['Ws_ptr'] = cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(Ws_layout)], 1024]
    SharedStorage.__annotations__['Vs_ptr'] = cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(Vs_layout)], 1024]
    SharedStorage.__annotations__['Xs_ptr'] = cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cute.cosize(Xs_layout)], 1024]
    SharedStorage.__annotations__['pipew_ptr'] = cute.struct.MemRange[cutlass.Int64, self.stages * 2]
    SharedStorage.__annotations__['pipev_ptr'] = cute.struct.MemRange[cutlass.Int64, self.stages * 2]
    SharedStorage.__annotations__['pipex_ptr'] = cute.struct.MemRange[cutlass.Int64, self.stages * 2]
    smem_allocator_ = cutlass.utils.SmemAllocator()
    smem__ = smem_allocator_.allocate(cute.struct(SharedStorage))
    Ws = smem__.Ws_ptr.get_tensor(Ws_layout.outer, swizzle=Ws_layout.inner)
    Vs = smem__.Vs_ptr.get_tensor(Vs_layout.outer, swizzle=Vs_layout.inner)
    Xs = smem__.Xs_ptr.get_tensor(Xs_layout.outer, swizzle=Xs_layout.inner)

    # Make Pipelines
    prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
    cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=self.num_consumer_warps)
    pipew = pipeline.PipelineTmaAsync.create(
      barrier_storage=smem__.pipew_ptr.data_ptr(),
      num_stages=self.stages,
      producer_group=prod_grp,
      consumer_group=cons_grp,
      tx_count=cute.size_in_bytes(cutlass.BFloat16, cute.select(Ws_layout, mode=[0, 1])),
      defer_sync=False,
    )
    pipev = pipeline.PipelineTmaAsync.create(
      barrier_storage=smem__.pipev_ptr.data_ptr(),
      num_stages=self.stages,
      producer_group=prod_grp,
      consumer_group=cons_grp,
      tx_count=cute.size_in_bytes(cutlass.BFloat16, cute.select(Vs_layout, mode=[0, 1])),
      defer_sync=False,
    )
    pipex = pipeline.PipelineTmaAsync.create(
      barrier_storage=smem__.pipex_ptr.data_ptr(),
      num_stages=self.stages,
      producer_group=prod_grp,
      consumer_group=cons_grp,
      tx_count=cute.size_in_bytes(cutlass.BFloat16, cute.select(Vs_layout, mode=[0, 1])),
      defer_sync=False,
    )

    scheduler = Gemm2DTileScheduler.create(scheduler_params)
    if (warp_idx < self.num_consumer_warps): # [None, 8) --> (8 warps)
      # normalized_warp_idx unchanged
      # tidx unchanged
      cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
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
    if (self.num_consumer_warps <= warp_idx < (self.num_consumer_warps + self.num_producer_warps)): # [8, (8 + 4)) --> (4 warps)
      normalized_warp_idx = warp_idx - self.num_consumer_warps
      tidx = tidx - (8 * cute.arch.WARP_SIZE)
      pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
      w2_work_tile = scheduler.initial_work_tile_info()
      while w2_work_tile.is_valid_tile:
        w2_tile_coord = w2_work_tile.tile_idx
        X_g2s_tma_tensor_slice = X_g2s_tma_tensor[None, None, w2_tile_coord[3]]
        V_g2s_tma_tensor_slice = V_g2s_tma_tensor[None, None, w2_tile_coord[3]]
        W_g2s_tma_tensor_slice = W_g2s_tma_tensor[None, None, w2_tile_coord[3]]
        if normalized_warp_idx == 0:
            # normalized_warp_idx unchanged
            # tidx unchanged
            for k1 in cutlass.range(4, unroll=1):
                pipex.producer_acquire(pstate)
                shared.tma_copy(X_g2s_tma_atom, X_g2s_tma_tensor_slice, Xs, 128, 64, w2_tile_coord[0], k1, pipex, pstate)
                pipev.producer_acquire(pstate)
                shared.tma_copy(V_g2s_tma_atom, V_g2s_tma_tensor_slice, Vs, 128, 64, w2_tile_coord[1], k1, pipev, pstate)
                pipew.producer_acquire(pstate)
                shared.tma_copy(W_g2s_tma_atom, W_g2s_tma_tensor_slice, Ws, 128, 64, w2_tile_coord[1], k1, pipew, pstate)
        scheduler.fetch_next_work()
        scheduler.advance_to_next_work()
        w2_work_tile = scheduler.get_current_work()