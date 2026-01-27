import argparse
from typing import Callable, Tuple, Type
import cuda.bindings.driver as cuda

import torch
from triton import runtime
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

from pipeline import PipelineTmaAsync

from tile_scheduler import SingleTileScheduler, TileSchedulerArguments
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
def printwg(x):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        if tidx%128 == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.print_tensor(x)
    else:
        if tidx%128 == 0 and bidx == 0 and bidy == 0 and bidz == 0:
            cute.printf(x)

class FlashSM90:
    def __init__(
        self,
        dtype,
        qk_mn: Tuple[int, int],
        cluster_size_m: int=1,
        num_stages: int=2,
        ):
        self.dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.num_stages = num_stages
        self.tile_m, self.tile_n = qk_mn
        self.hdimv = 0 # TODO
        self.hdimk = 0 # TODO
        self.mnk_qk = (self.tile_m, self.tile_n, self.hdimk)
        self.buffer_align_bytes = 1024
        self.is_mcast = cluster_size_m > 1
        self.num_mcast = cluster_size_m

        # compile time, later
        self.num_mma_threads = None
        self.num_mma_warpgroups = None
        self.num_mma_regs = None
        self.num_producer_regs = None
        self.tma_copy_bytes = None

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
                stream: cuda.CUstream):
        self.hdimk = cute.size(mQ, mode=[3])
        self.hdimv = cute.size(mV, mode=[3])
        
        mQ, mK, mV, mO = [my_utils.select(x, [2, 3, 1, 0]) for x in (mQ, mK, mV, mO)] # (b, h, seqlen, d) --> (seqlen, d, h, b)
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_warpgroups = self.num_mma_threads / THREADS_PER_WG
        self.num_threads = int((self.num_mma_warpgroups + 1) * THREADS_PER_WG)

        assert self.num_mma_warpgroups in (1, 2, 3)

        # self.num_mma_regs = (256, 240, 160)[self.num_mma_warpgroups - 1]
        # self.num_producer_regs = (56, 24, 32)[self.num_mma_warpgroups - 1]
        self.num_mma_regs = 232
        self.num_producer_regs = 40 # allows you to debug print

        # make smem layout
        self._get_smem_layouts()
        self._get_shared_storage_cls()
        (tma_atom_q, tma_tensor_q, 
         tma_atom_k, tma_tensor_k, 
         tma_atom_v, tma_tensor_v, 
         tma_atom_o, tma_tensor_o) = self._get_tma_copy_attrs(mQ, mK, mV, mO)
        n_block_max = cute.ceil_div(cute.size(mK, mode=[0]), self.tile_n)
        
        # BlockInfo tells each block where to start/stop iterating, but we'll skip it since we're doing full non-causal attention
        # they also have seqlen to store seqlen info, no clue what that is w.r.t. lol but it loads everything once instead of over+over

        # Tile Scheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m), # n blocks
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            (self.num_mcast, 1),
        )
        tile_sched_params = SingleTileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = SingleTileScheduler.get_grid_shape(tile_sched_params)
        # LOG2_E = math.log2(math.e) # I think we can multiply by 1/sqrt(d) here too

        # we should be ready to do the kernel now
        print("launching kernel")
        self.kernel(tma_tensor_q, tma_tensor_k, tma_tensor_v, tma_tensor_o, tma_atom_q, tma_atom_k, tma_atom_v, tma_atom_o, self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout, tiled_mma_qk, tiled_mma_pv, n_block_max, SingleTileScheduler, tile_sched_params).launch(grid=grid_dim, block=[self.num_threads, 1, 1], cluster=[self.num_mcast, 1, 1], stream=stream)
    
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, 
               tma_atom_q: cute.CopyAtom, tma_atom_k: cute.CopyAtom, tma_atom_v: cute.CopyAtom, tma_atom_o: cute.CopyAtom,
               sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout, sV_layout: cute.ComposedLayout, sO_layout: cute.ComposedLayout,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               n_block_max: int,
               TileScheduler: cutlass.Constexpr[Callable], tile_sched_params: ParamsBase):
        """
        First wg(0-4) --> producer
        other wg --> consumer
        """
        print0("entered kernel")
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            for tma_atom in (tma_atom_q,):
                # if const_expr(tma_atom is not None)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)
        
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_Q = storage.mbar_qo.data_ptr() # TODO need to init?
        pipeline_kv_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
        pipeline_kv_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE)
        pipeline_k = PipelineTmaAsync.create(
            barrier_storage=storage.mbar_k.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["K"], # should add tma copy bytes [k]
            cta_layout_vmnk=cute.make_layout((1, self.num_mcast, 1, 1)),
            defer_sync=True,
        )
        pipeline_v = PipelineTmaAsync.create(
            barrier_storage=storage.mbar_v.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
            cta_layout_vmnk=cute.make_layout((1, self.num_mcast, 1, 1)),
            defer_sync=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner) # (n, dimv) --> (k, n) in mnk
        sVt = my_utils.transpose_view(sV) # (dimv, n) --> (n, k)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype) # reuse sQ
        # need to know how many blocks to iterate through here

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        print0("at branching")
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
            self.mma(n_block_max, sQ, sK, sVt, sO, mO, pipeline_k, pipeline_v, mma_qk, mma_pv, tma_atom_o, tidx, TileSchedulerCls)
    
    @cute.jit
    def load(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, 
             sQ: cute.Tensor, sK: cute.Tensor, sV: cute.Tensor, 
             tma_atom_q: cute.CopyAtom, tma_atom_k: cute.CopyAtom, tma_atom_v: cute.CopyAtom, 
             pipeline_k: pipeline.PipelineAsync, pipeline_v: pipeline.PipelineAsync, n_block_max: int, TileSchedulerCls: Callable):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # cta_layout = cute.make_layout((self.num_mcast, 1))
        # do I even need an mcast mask?
        if warp_idx_in_wg == 0:
            kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            m_block, head_idx, batch_idx = work_tile.tile_idx
            mQ_curr = mQ[None, None, head_idx, batch_idx]
            mK_curr = mK[None, None, head_idx, batch_idx]
            mV_curr = mV[None, None, head_idx, batch_idx]

            gQ = cute.local_tile(mQ_curr, (self.tile_m, self.hdimk), (m_block, 0))
            gK = cute.local_tile(mK_curr, (self.tile_n, self.hdimk), (None, 0))
            gV = cute.local_tile(mV_curr, (self.tile_n, self.hdimv), (None, 0))
            
            load_Q, _, _ = my_utils.tma_get_copy_fn(
                tma_atom_q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )

            load_K, _, _ = my_utils.tma_get_copy_fn(
                tma_atom_k, cta_rank_in_cluster, cute.make_layout(1), gK, sK, 
            )

            load_V, _, _ = my_utils.tma_get_copy_fn(
                tma_atom_v, cta_rank_in_cluster, cute.make_layout(1), gV, sV,
            )

            # 1824: First pipeline K, just add Q in there
            n_block = n_block_max - 1
            pipeline_k.producer_acquire(
                kv_producer_state,
                extra_tx_count=self.tma_copy_bytes["Q"]
            )
            load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))
            load_K(n_block, kv_producer_state.index, tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))

            pipeline_v.producer_acquire(kv_producer_state)
            load_V(n_block, kv_producer_state.index, tma_bar_ptr=pipeline_v.producer_get_barrier(kv_producer_state))
            kv_producer_state.advance()

            # this is no intra wg overlap
            for i in cutlass.range(n_block_max - 1, unroll=1):
                n_block = n_block_max - 2 - i
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(n_block, kv_producer_state.index, tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(n_block, kv_producer_state.index, tma_bar_ptr=pipeline_v.producer_get_barrier(kv_producer_state))
                kv_producer_state.advance()
    
    @cute.jit
    def mma(self, n_block_max: int, 
            sQ: cute.Tensor, sK: cute.Tensor, sVt: cute.Tensor, sO: cute.Tensor,
            mO: cute.Tensor,
            pipeline_k: pipeline.PipelineAsync, pipeline_v: pipeline.PipelineAsync, mma_qk: cute.TiledMma, mma_pv: cute.TiledMma, tma_atom_o: cute.CopyAtom, tidx: Int32, TileSchedulerCls: Callable):
        kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
        # pipeline_k.consumer_wait(kv_consumer_state)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx = work_tile.tile_idx

        thr_mma_qk = mma_qk.get_slice(tidx)
        tSrQ = mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        acc_p_shape = mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        tOrP = cute.make_rmem_tensor(my_utils.convert_layout_acc_frgA(cute.make_layout(acc_p_shape)), self.dtype) # we'll copy p_acc into here later
        # look at how they declare tOrP, to feed into PVGemm

        thr_mma_pv = mma_pv.get_slice(tidx) # this differs from what FA does but it should work
        tOrVt = mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))
        acc_o_shape = mma_pv.partition_shape_C((self.tile_m, self.hdimv))
        acc_o = cute.make_rmem_tensor(acc_o_shape, self.acc_dtype)
        O_should_accumulate = False
        for n_tile in cutlass.range(n_block_max, unroll=1):
            pipeline_k.consumer_wait(kv_consumer_state)
            # Qk stuff
            p_acc = my_utils.gemm_zero_init(mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK, B_idx=kv_consumer_state.index, wg_wait=-1)
            cute.nvgpu.warpgroup.wait_group(0)
            pipeline_k.consumer_release(kv_consumer_state)

            # TODO softmax here, they have a softmax.online_softmax
            # they call PTX directly to convert to f16 damnn
            tOrP_acc = cute.make_tensor(p_acc.iterator, my_utils.convert_layout_acc_frgA(p_acc.layout))
            tOrP.store(tOrP_acc.load().to(self.dtype))

            pipeline_v.consumer_wait(kv_consumer_state)
            # mma pv
            my_utils.gemm_w_index(mma_pv, acc_o, tOrP, tOrVt, not O_should_accumulate, B_idx=kv_consumer_state.index, wg_wait=0)
            O_should_accumulate = True
            pipeline_v.consumer_release(kv_consumer_state)
            kv_consumer_state.advance()
        # TODO need epilogue here now, then we can test correctness with doublegemm
        self.epilogue(acc_o, sO, mO, tma_atom_o, mma_pv, tidx, m_block, head_idx, batch_idx)
    
    @cute.jit
    def epilogue(self, acc_o: cute.Tensor, sO: cute.Tensor, mO: cute.Tensor, tma_atom_O: cute.CopyAtom, tiled_mma: cute.TiledMma, tidx: Int32, m_block: int, head_idx: int, batch_idx: int):
        rO = cute.make_fragment_like(acc_o, self.dtype)
        rO.store(acc_o.load().to(self.dtype))

        # make sure V is read
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads)
        
        # I think they reuse V memory for O since you can start loading the next q and k? not sure
        # copy rO to sO
        # mO_cur = mO[None, None, head_idx, batch_idx]
        # they use a barrier after stmatrix to wait for everything to be avail, then warp 4(first consumer does TMA store)
        # TMA out
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

        # extra +WARP_SIZE because warp 4 will arrive again before doing the tma store
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE
            )
            # store...
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
        # This differs from what they do, you can look at their flash/hopper_helpers.py for details

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

        mbar_ptr_QO = cute.struct.MemRange[cutlass.Int64, 2] # mbar for q and o
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        # I think we reuse Q for O? Not sure
        @cute.struct
        class SharedStorage:
            mbar_qo: mbar_ptr_QO
            mbar_k: mbar_ptr_K_struct
            mbar_v: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
        
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


if __name__ == "__main__":
    q = torch.randn((2, 2, 1024, 64), dtype=torch.bfloat16).add(0.01).to('cuda')
    k = torch.randn((2, 2, 1024, 64), dtype=torch.bfloat16).add(0.01).to('cuda')
    v = torch.randn((2, 2, 1024, 64), dtype=torch.bfloat16).add(0.01).to('cuda')
    o = torch.zeros((2, 2, 1024, 64), dtype=torch.bfloat16).to('cuda')

    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
    )
    [q_cute, k_cute, v_cute, o_cute] = [convert_from_dlpack(x) for x in (q, k, v, o)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    fa = FlashSM90(dtype=cutlass.BFloat16, qk_mn=(128, 256), cluster_size_m=1)
    fa(q_cute, k_cute, v_cute, o_cute, current_stream)

    ref = (q @ k.transpose(2, 3)) @ v
    # print(o)
    n_incorrect = o.numel() - ((o - ref).abs() < 0.01).sum().item()
    print('allclose:', torch.allclose(ref, o, atol=1e-1, rtol=1e-2)) # look at docs for torch.testing.assert_close for details
    print('max error:', torch.max((o-ref).abs()).item())
    print(f'{n_incorrect=}')