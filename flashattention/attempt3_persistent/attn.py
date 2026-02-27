import argparse
from typing import Callable, Tuple, Type
import cuda.bindings.driver as cuda

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton import runtime
from triton.testing import do_bench
import functools
import statistics

import cutlass
from cutlass import Boolean, Int32, const_expr
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait, PipelineState, PipelineUserType
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
from my_softmax import Softmax

from pipeline import PipelineTmaAsync

from tile_scheduler import StaticPersistentScheduler, SingleTileScheduler, TileSchedulerArguments
from cute_dsl_utils import ParamsBase
from functools import partial
import my_utils
import math
import enum

torch.manual_seed(42)

THREADS_PER_WG = 128

class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()

@cute.jit
def print0(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

@cute.jit
def printc(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx == 128 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx == 128 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

def get_tflops(bs, nh, lq, lkv, head_dim, head_dim_v, latency_ms):
    qk = bs * nh * (2 * lq * lkv * head_dim)
    smx = bs * nh * (4 * lq * lkv) # max + sub + exp + sum + div, but codebases(e.g. thunderkittens) use 4
    kv = bs * nh * (2 * lq * lkv * head_dim_v)
    return (qk + smx + kv) / latency_ms / 1e9

class FlashSM90:
    def __init__(
        self,
        qk_mn: Tuple[int, int],
        cluster_size_m: int=1,
        num_stages: int=2,
        intra_wg_overlap: bool=False,
        pingpong: bool=False,
        ):
        self.acc_dtype = cutlass.Float32
        self.num_stages = num_stages
        self.tile_m, self.tile_n = qk_mn
        self.buffer_align_bytes = 1024
        self.is_mcast = cluster_size_m > 1
        self.num_mcast = cluster_size_m
        self.intra_wg_overlap = intra_wg_overlap
        self.pingpong = pingpong

        # compile time, later
        self.dtype = None
        self.num_mma_threads = None
        self.num_mma_warpgroups = None
        self.num_mma_regs = None
        self.num_producer_regs = None
        self.tma_copy_bytes = None

        # FA assumes dims are small(<512), so we can single-gemm across these dims
        # We want <256 since we don't support (1, 2) MMA atoms yet
        self.hdimv = None
        self.hdimk = None

        self.sQ_layout = None
        self.sK_layout = None
        self.sV_layout = None
        self.sO_layout = None
        self.shared_storage = None

    @cute.jit
    def __call__(
                self, 
                mQ: cute.Tensor,
                mK: cute.Tensor,
                mV: cute.Tensor,
                mO: cute.Tensor,
                softmax_scale: cutlass.Float32, # 1/sqrt(D)
                stream: cuda.CUstream):
        self.dtype = mQ.element_type
        self.hdimk = cute.size(mQ, mode=[3])
        self.hdimv = cute.size(mV, mode=[3])
        
        mQ, mK, mV, mO = [my_utils.select(x, [2, 3, 1, 0]) for x in (mQ, mK, mV, mO)] # (b, h, seqlen, d) --> (seqlen, d, h, b)
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_warpgroups = self.num_mma_threads / THREADS_PER_WG
        self.num_threads = int((self.num_mma_warpgroups + 1) * THREADS_PER_WG)

        assert self.num_mma_warpgroups in (1, 2, 3)

        self.num_mma_regs = (256, 240, 160)[int(self.num_mma_warpgroups - 1)]
        self.num_producer_regs = (56, 24, 32)[int(self.num_mma_warpgroups - 1)]
        
        # allows you to debug print
        self.num_mma_regs = 232
        self.num_producer_regs = 40

        # Shared Storage
        self._get_smem_layouts()
        self._get_shared_storage_cls()
        (tma_atom_q, tma_tensor_q, 
         tma_atom_k, tma_tensor_k, 
         tma_atom_v, tma_tensor_v, 
         tma_atom_o, tma_tensor_o) = self._get_tma_copy_attrs(mQ, mK, mV, mO)
        n_block_max = cute.ceil_div(cute.size(mK, mode=[0]), self.tile_n)

        # Tile Scheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m), # n blocks
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            (self.num_mcast, 1),
        )
        tile_sched_params = StaticPersistentScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = StaticPersistentScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        self.kernel(
            tma_tensor_q, tma_tensor_k, tma_tensor_v, tma_tensor_o, 
            tma_atom_q, tma_atom_k, tma_atom_v, tma_atom_o, 
            self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout, 
            tiled_mma_qk, tiled_mma_pv, 
            n_block_max, softmax_scale_log2, 
            StaticPersistentScheduler, tile_sched_params).launch(grid=grid_dim, block=[self.num_threads, 1, 1], cluster=[self.num_mcast, 1, 1], stream=stream, min_blocks_per_mp=1)
    
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, 
               tma_atom_q: cute.CopyAtom, tma_atom_k: cute.CopyAtom, tma_atom_v: cute.CopyAtom, tma_atom_o: cute.CopyAtom,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout, sO_layout: cute.ComposedLayout,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               n_block_max: int,
               softmax_scale_log2: cutlass.Float32,
               TileScheduler: cutlass.Constexpr[Callable], tile_sched_params: ParamsBase):
        """
        First wg(0-4) --> producer
        other wg --> consumer
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == 0:
            for tma_atom in (tma_atom_q, tma_atom_v, tma_atom_o):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        pipeline_kv_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
        pipeline_kv_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, (self.num_mma_threads // cute.arch.WARP_SIZE) * self.num_mcast)

        pipeline_k = PipelineTmaAsync.create(
            barrier_storage=storage.mbar_k.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cute.make_layout((1, self.num_mcast, 1, 1)),
            defer_sync=False,
        )
        pipeline_v = PipelineTmaAsync.create(
            barrier_storage=storage.mbar_v.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
            cta_layout_vmnk=cute.make_layout((1, self.num_mcast, 1, 1)),
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner) # (n, dimv) --> (k, n) in mnk
        sVt = my_utils.transpose_view(sV) # (dimv, n) --> (n, k). Required for GEMM to work
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype) # reuse sQ

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ, mK, mV, 
                sQ, sK, sV, 
                tma_atom_q, tma_atom_k, tma_atom_v, 
                pipeline_k, pipeline_v,
                n_block_max, TileSchedulerCls)
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            self.mma(n_block_max, sQ, sK, sVt, sO, mO, pipeline_k, pipeline_v, mma_qk, mma_pv, tma_atom_o, tidx, softmax_scale_log2, TileSchedulerCls)
    
    @cute.jit
    def load(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, 
             sQ: cute.Tensor, sK: cute.Tensor, sV: cute.Tensor, 
             tma_atom_q: cute.CopyAtom, tma_atom_k: cute.CopyAtom, tma_atom_v: cute.CopyAtom, 
             pipeline_k: pipeline.PipelineAsync, pipeline_v: pipeline.PipelineAsync, n_block_max: int, TileSchedulerCls: Callable):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cta_layout = cute.make_layout((self.num_mcast, 1))
        mcast_mask = cute.make_layout_image_mask(
            cta_layout, (cta_rank_in_cluster, 0), mode=0
        )

        if warp_idx_in_wg == 0:
            k_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            v_producer_state = k_producer_state.clone()            
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx = work_tile.tile_idx
                mQ_curr = mQ[None, None, head_idx, batch_idx] # (seqlen, dim)
                mK_curr = mK[None, None, head_idx, batch_idx]
                mV_curr = mV[None, None, head_idx, batch_idx]

                gQ = cute.local_tile(mQ_curr, (self.tile_m, self.hdimk), (m_block, 0))
                gK = cute.local_tile(mK_curr, (self.tile_n, self.hdimk), (None, 0))
                gV = cute.local_tile(mV_curr, (self.tile_n, self.hdimv), (None, 0))
                
                load_Q, _, _ = my_utils.tma_get_copy_fn(
                    tma_atom_q, 0, cute.make_layout(1), gQ, sQ, single_stage=True, mcast_mask=0
                )

                load_K, _, _ = my_utils.tma_get_copy_fn(
                    tma_atom_k, 0, cute.make_layout(1), gK, sK, mcast_mask=mcast_mask
                )

                load_V, _, _ = my_utils.tma_get_copy_fn(
                    tma_atom_v, 0, cute.make_layout(1), gV, sV, mcast_mask=mcast_mask
                )

                if cutlass.const_expr(self.intra_wg_overlap):
                    # Initial QK
                    n_block = n_block_max - 1
                    pipeline_k.producer_acquire(
                        k_producer_state,
                        extra_tx_count=self.tma_copy_bytes["Q"]
                    )
                    load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state))
                    load_K(n_block, k_producer_state.index, tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state))
                    k_producer_state.advance()

                    # K[next] and V[curr]
                    for i in cutlass.range(n_block_max - 1, unroll=1):
                        n_block = n_block_max - 1 - i
                        pipeline_k.producer_acquire(k_producer_state)
                        load_K(n_block - 1, k_producer_state.index, tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state))
                        k_producer_state.advance()
                        pipeline_v.producer_acquire(v_producer_state)
                        load_V(n_block, v_producer_state.index, tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state))
                        v_producer_state.advance()
                    
                    # last V load
                    pipeline_v.producer_acquire(v_producer_state)
                    load_V(0, v_producer_state.index, tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state))
                    v_producer_state.advance()
                if cutlass.const_expr(not self.intra_wg_overlap):
                    # 1824: Add Q to pipeline K for first iter, no need for additional barrier
                    n_block = n_block_max - 1
                    pipeline_k.producer_acquire(
                        k_producer_state,
                        extra_tx_count=self.tma_copy_bytes["Q"]
                    )
                    load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state))
                    load_K(n_block, k_producer_state.index, tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state))

                    pipeline_v.producer_acquire(v_producer_state)
                    load_V(n_block, v_producer_state.index, tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state))
                    k_producer_state.advance()
                    v_producer_state.advance()

                    # no intra wg overlap
                    for i in cutlass.range(n_block_max - 1, unroll=1):
                        n_block = n_block_max - 2 - i
                        pipeline_k.producer_acquire(k_producer_state)
                        load_K(n_block, k_producer_state.index, tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state))
                        pipeline_v.producer_acquire(v_producer_state)
                        load_V(n_block, v_producer_state.index, tma_bar_ptr=pipeline_v.producer_get_barrier(v_producer_state))
                        k_producer_state.advance()
                        v_producer_state.advance()
                cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
            pipeline_k.producer_tail(k_producer_state)
            pipeline_v.producer_tail(v_producer_state)
    
    @cute.jit
    def mma(self, n_block_max: int, 
            sQ: cute.Tensor, sK: cute.Tensor, sVt: cute.Tensor, sO: cute.Tensor,
            mO: cute.Tensor,
            pipeline_k: pipeline.PipelineAsync, pipeline_v: pipeline.PipelineAsync, mma_qk: cute.TiledMma, mma_pv: cute.TiledMma, tma_atom_o: cute.CopyAtom, tidx: Int32, softmax_scale_log2: cutlass.Float32, TileSchedulerCls: Callable):
        kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        thr_mma_qk = mma_qk.get_slice(tidx)
        tSrQ = mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        acc_p_shape = mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        tOrP = cute.make_rmem_tensor(my_utils.convert_layout_acc_frgA(cute.make_layout(acc_p_shape)), self.dtype)

        thr_mma_pv = mma_pv.get_slice(tidx)
        tOrVt = mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))
        acc_o_shape = mma_pv.partition_shape_C((self.tile_m, self.hdimv))
        acc_o = cute.make_rmem_tensor(acc_o_shape, self.acc_dtype)

        softmax = Softmax.create(softmax_scale_log2, num_rows=acc_o.shape[0][0] * acc_o.shape[1])
        # no need to softmax.reset, calling with first_block takes care of it
        
        self.inter_wg_iwo_init_barrier()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            if cutlass.const_expr(self.intra_wg_overlap):
                kv_consumer_state = self.first_half_block_overlap(mma_qk, tSrQ, tSrK, tOrP, pipeline_k, kv_consumer_state, softmax)
                O_should_accumulate = False
                for _ in cutlass.range(n_block_max-1, unroll=1):
                    kv_consumer_state = self.mma_one_n_block_iwo(pipeline_k, pipeline_v, kv_consumer_state, mma_qk, tSrQ, tSrK, softmax, tOrP, acc_o, mma_pv, tOrVt, O_should_accumulate, False)
                    O_should_accumulate = True
                kv_consumer_state = self.last_half_block_overlap(acc_o, tOrP, tOrVt, kv_consumer_state, pipeline_v, mma_pv)
            if cutlass.const_expr(not self.intra_wg_overlap):
                self.inter_wg_barrier() # all consumers sync before starting
                kv_consumer_state = self.mma_one_n_block(pipeline_k, pipeline_v, kv_consumer_state, mma_qk, tSrQ, tSrK, softmax, tOrP, acc_o, mma_pv, tOrVt, False, True, sQ, sK)
                for _ in cutlass.range(n_block_max-1, unroll=1):
                    kv_consumer_state = self.mma_one_n_block(pipeline_k, pipeline_v, kv_consumer_state, mma_qk, tSrQ, tSrK, softmax, tOrP, acc_o, mma_pv, tOrVt, True, False, sQ, sK)
                self.inter_wg_arrive()
            row_scale = softmax.finalize()
            softmax.rescale_O(acc_o, row_scale)

            self.epilogue(acc_o, sO, mO, tma_atom_o, mma_pv, tidx, m_block, head_idx, batch_idx)
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
    
    @cute.jit
    def first_half_block_overlap(self, mma_qk: cute.TiledMma, tSrQ: cute.Tensor, tSrK: cute.Tensor, tOrP: cute.Tensor, pipeline_k: pipeline.PipelineAsync, kv_consumer_state: pipeline.PipelineState, softmax: Softmax):
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        p_acc = my_utils.gemm_zero_init(mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK, B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        softmax.online_softmax(p_acc, is_first=True) # no need to rescale on first iter
        tOrP_acc = cute.make_tensor(p_acc.iterator, my_utils.convert_layout_acc_frgA(p_acc.layout))
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        my_utils.cvt_f16(tOrP_acc, tOrP)

        return kv_consumer_state
    
    @cute.jit
    def mma_one_n_block(self, 
                        pipeline_k: pipeline.PipelineAsync, pipeline_v: pipeline.PipelineAsync, 
                        kv_consumer_state: pipeline.PipelineState, 
                        mma_qk: cute.TiledMma, tSrQ: cute.Tensor, tSrK: cute.Tensor, 
                        softmax: Softmax, 
                        tOrP: cute.Tensor,
                        acc_o: cute.Tensor,
                        mma_pv: cute.TiledMma,
                        tOrVt: cute.Tensor,
                        O_should_accumulate: Boolean,
                        softmax_is_first: cutlass.Constexpr[bool], sQ: cute.Tensor, sK: cute.Tensor):
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        
        # QKGemm
        p_acc = my_utils.gemm_zero_init(mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK, B_idx=kv_consumer_state.index, wg_wait=-1)
        self.inter_wg_arrive()
        cute.nvgpu.warpgroup.wait_group(0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Softmax
        # TODO they call PTX directly to convert to f16 damnn
        row_scale = softmax.online_softmax(p_acc, is_first=softmax_is_first) # rescale p_acc, internally store sum/max
        tOrP_acc = cute.make_tensor(p_acc.iterator, my_utils.convert_layout_acc_frgA(p_acc.layout))
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        my_utils.cvt_f16(tOrP_acc, tOrP)
        softmax.rescale_O(acc_o, row_scale)

        # PVGemm
        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        self.inter_wg_barrier()
        my_utils.gemm_w_index(mma_pv, acc_o, tOrP, tOrVt, not O_should_accumulate, B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block_iwo(self, 
                        pipeline_k: pipeline.PipelineAsync, pipeline_v: pipeline.PipelineAsync, 
                        kv_consumer_state: pipeline.PipelineState, 
                        mma_qk: cute.TiledMma, tSrQ: cute.Tensor, tSrK: cute.Tensor, 
                        softmax: Softmax, 
                        tOrP: cute.Tensor,
                        acc_o: cute.Tensor,
                        mma_pv: cute.TiledMma,
                        tOrVt: cute.Tensor,
                        O_should_accumulate: Boolean,
                        softmax_is_first: cutlass.Constexpr[bool]):
        v_consumer_state = kv_consumer_state.clone()
        kv_consumer_state.advance() # v is one behind kv state

        # QKGemm[next]
        # this creates extra registers for to hold p_acc and tOrP simultaneously
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        self.inter_wg_barrier()
        p_acc = my_utils.gemm_zero_init(mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK, B_idx=kv_consumer_state.index, wg_wait=-1)

        # PVGemm[current]
        pipeline_v.consumer_wait(v_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        my_utils.gemm_w_index(mma_pv, acc_o, tOrP, tOrVt, not O_should_accumulate, B_idx=v_consumer_state.index, wg_wait=-1)

        self.inter_wg_arrive()
        cute.nvgpu.warpgroup.wait_group(1) # QK[next] done
        pipeline_k.consumer_release(kv_consumer_state)

        # Softmax
        # TODO they call PTX directly to convert to f16 damnn
        row_scale = softmax.online_softmax(p_acc, is_first=False) # rescale p_acc, internally store sum/max

        # Need PVGemm[current] to finish before rescaling O for next iter
        cute.nvgpu.warpgroup.wait_group(0)
        pipeline_v.consumer_release(v_consumer_state)

        tOrP_acc = cute.make_tensor(p_acc.iterator, my_utils.convert_layout_acc_frgA(p_acc.layout))
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        my_utils.cvt_f16(tOrP_acc, tOrP)
        softmax.rescale_O(acc_o, row_scale)

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(self, acc_o: cute.Tensor, tOrP: cute.Tensor, tOrVt: cute.Tensor, kv_consumer_state: pipeline.PipelineState, pipeline_v: pipeline.PipelineAsync, mma_pv: cute.TiledMma):
        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        my_utils.gemm_w_index(mma_pv, acc_o, tOrP, tOrVt, False, B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state
    
    @cute.jit
    def inter_wg_iwo_init_barrier(self):
        warp_group_idx = my_utils.canonical_warp_group_idx(sync=False)
        if cutlass.const_expr(self.pingpong):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * THREADS_PER_WG
                )

    @cute.jit
    def inter_wg_barrier(self):
        if cutlass.const_expr(self.pingpong):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) - 1 + my_utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * THREADS_PER_WG,
            )
    
    @cute.jit
    def inter_wg_arrive(self):
        if cutlass.const_expr(self.pingpong):
            assert self.num_mma_warpgroups in [2, 3]
            cur_wg = my_utils.canonical_warp_group_idx(sync=False) - 1
            if cutlass.const_expr(self.num_mma_warpgroups == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_mma_warpgroups
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * THREADS_PER_WG
            )

    
    @cute.jit
    def epilogue(self, acc_o: cute.Tensor, sO: cute.Tensor, mO: cute.Tensor, tma_atom_O: cute.CopyAtom, tiled_mma: cute.TiledMma, tidx: Int32, m_block: int, head_idx: int, batch_idx: int):
        # convert down IN REGISTERS
        rO = cute.make_fragment_like(acc_o, self.dtype)
        rO.store(acc_o.load().to(self.dtype))

        # Make sure no SMEM dependencies
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)

        # Copy R2S
        smem_copy_atom_O = my_utils.get_smem_store_atom(90, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taco = smem_thr_copy_O.retile(rO)
        taco_s = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taco, taco_s)

        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE
        )

        mO_curr = mO[None, None, head_idx, batch_idx]
        gO = cute.local_tile(mO_curr, (self.tile_m, self.hdimv), (m_block, 0))

        store_O, _, _ = my_utils.tma_get_copy_fn(
            tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
        )

        # extra +WARP_SIZE because warp 4 will arrive again before doing the tma store.
        # Barrier ensures everything is in SMEM
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE
            )
            # TMA store
            store_O()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True) # .read: no need to wait for writes to finish, just finish reading from SMEM
        
    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )

        tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.warpgroup.OperandMajorMode.K,
            cute.nvgpu.warpgroup.OperandMajorMode.MN,
            cutlass.Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.hdimv),
            a_source=cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv
    
    def _get_smem_layouts(self):
        q_smem_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, self.dtype, self.hdimk),
            self.dtype
        )
        k_smem_atom = q_smem_atom
        
        v_smem_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, self.dtype, self.hdimv),
            self.dtype
        )
        o_smem_atom = v_smem_atom

        self.sQ_layout = cute.tile_to_shape(
            q_smem_atom, (self.tile_m, self.hdimk), (0, 1)
        )
        self.sK_layout = cute.tile_to_shape(
            k_smem_atom, (self.tile_n, self.hdimk, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            v_smem_atom, (self.tile_n, self.hdimv, self.num_stages),
            (0, 1, 2),
        )

        self.sO_layout = cute.tile_to_shape(
            o_smem_atom, (self.tile_m, self.hdimv), (0, 1)
        )
    
    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        sO_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sO_layout)], self.buffer_align_bytes]

        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorage:
            mbar_q: mbar_ptr_Q_struct
            mbar_k: mbar_ptr_K_struct
            mbar_v: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sO: sO_struct
        
        self.shared_storage = SharedStorage
    
    def _get_tma_copy_attrs(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor):
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(x.element_type, cute.select(layout, mode=[0, 1]))
            for name, x, layout in [
                ('Q', mQ, self.sQ_layout),
                ('K', mK, self.sK_layout),
                ('V', mV, self.sV_layout),
            ]
        }

        gcq = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        gckv = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() if not self.is_mcast else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        gso = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            gcq,
            mQ,
            self.sQ_layout,
            (self.tile_m, self.hdimk),
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            gckv,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.hdimk),
            self.num_mcast
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            gckv,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.hdimv),
            self.num_mcast
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            gso,
            mO,
            self.sO_layout,
            (self.tile_m, self.hdimv)
        )
        return tma_atom_q, tma_tensor_q, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_o, tma_tensor_o

def attn_reimpl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    p = q @ k.transpose(2, 3)
    p = p * ((q.size(-1)**-0.5) * math.log2(math.e))
    p = torch.exp2(p - torch.max(p, dim=-1, keepdim=True).values)
    recip = torch.reciprocal(torch.sum(p, dim=-1, keepdim=True))
    pv = (p @ v)
    return pv * recip

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
    )

def profile_ms(op, repeats=30):
    stream = torch.cuda.current_stream()

    clear_cache = functools.partial(
        runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
        runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
    )
    clear_cache()

    # warmup
    op()
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        # clear_cache()
        start[i].record(stream)
        op()
        end[i].record(stream)

    torch.cuda.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in zip(start, end)])

if __name__ == "__main__":
    print("starting")
    bs, h = 4, 16
    dim = 128
    seqlen = 8192
    rt = 1 / math.sqrt(dim)
    q = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    k = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    v = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    o = torch.zeros((bs, h, seqlen, dim), dtype=torch.bfloat16).to('cuda')

    [q_cute, k_cute, v_cute, o_cute] = [convert_from_dlpack(x) for x in (q, k, v, o)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # good with dim=64
    # fa = FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=True, pingpong=True)
    
    # this actually beats cudnn on 4, 16, 8192, 128 
    fa = FlashSM90(qk_mn=(128, 128), num_stages=2, cluster_size_m=1, intra_wg_overlap=False, pingpong=True)
    compiled_fa = cute.compile(fa, q_cute, k_cute, v_cute, o_cute, rt, current_stream)
    compiled_fa = cute.compile(fa, q_cute, k_cute, v_cute, o_cute, rt, current_stream)
    compiled_fa(q_cute, k_cute, v_cute, o_cute, rt, current_stream)

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        time_torch = do_bench(lambda: F.scaled_dot_product_attention(q, k, v))
    time_ms = do_bench(lambda: compiled_fa(q_cute, k_cute, v_cute, o_cute, rt, current_stream), return_mode="median")
    # profile_ms(lambda: compiled_fa(q_cute, k_cute, v_cute, o_cute, 0.125, current_stream), repeats=30)

    ref = F.scaled_dot_product_attention(q, k, v)
    # ref = (q @ k.transpose(2, 3)) @ v
    n_incorrect = o.numel() - ((o - ref).abs() < 0.1).sum().item()
    allclose = torch.allclose(ref, o, atol=1e-1, rtol=1e-1)
    print('allclose:', allclose) # look at docs for torch.testing.assert_close for details
    if not allclose:
        print("!!!!!! WARNING -- INCORRECT !!!!!!")
    
    diff = (o - ref).abs()
    max_val, max_idx = diff.view(-1).max(0)
    bs, h, seqlen, dim = o.shape
    idx0 = max_idx // (h * seqlen * dim)
    idx1 = (max_idx % (h * seqlen * dim)) // (seqlen * dim)
    idx2 = (max_idx % (seqlen * dim)) // dim
    idx3 = max_idx % dim
    # print(ref[0, 0, idx2, ...])
    # print(o[0, 0, idx2, ...])

    print(f'Max error: {max_val.item()} at (bs, h, seqlen, dim) = ({idx0}, {idx1}, {idx2}, {idx3})')
    print(f'{n_incorrect=}')
    print(f'Mine:  {get_tflops(bs, h, seqlen, seqlen, dim, dim, time_ms)} TFLOPS ({time_ms})')
    print(f'Torch: {get_tflops(bs, h, seqlen, seqlen, dim, dim, time_torch)} TFLOPS ({time_torch})')
    print(f'{time_torch/time_ms}')