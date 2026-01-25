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

THREADS_PER_WG = 128

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
        cluster_size_m: int,
        num_stages: int=2,
        ):
        self.dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.num_stages = num_stages
        self.tile_m, self.tile_n = qk_mn
        self.hdimv = 64 # TODO
        self.hdimk = 64 # TODO
        self.mnk_qk = (self.tile_m, self.tile_n, self.hdimk)
        self.buffer_align_bytes = 1024

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
        print(mQ)
        
        # this just assumes strides must be divisible by 128 bits, but not sure why we do this
        new_stride = lambda t: (
            *(
                cute.assume(s, divby=128 // t.element_type.width)
                if not isinstance(s, int) or s != 0
                else s
                for s in t.stride[:-1]
            ),
            t.stride[-1],
        )
        # mQ, mK, mV, mO = [
        #     cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
        #     for t in (mQ, mK, mV, mO)
        # ]
        
        mQ, mK, mV, mO = [my_utils.select(x, [2, 3, 1, 0]) for x in (mQ, mK, mV, mO)] # (b, h, seqlen, d) --> (seqlen, d, h, b)
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
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
        
        # BlockInfo tells each block where to start/stop iterating, but we'll skip it since we're doing full non-causal attention
        # they also have seqlen to store seqlen info, no clue what that is w.r.t. lol but it loads everything once instead of over+over

        # Tile Scheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m), # n blocks
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
        )
        tile_sched_params = SingleTileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = SingleTileScheduler.get_grid_shape(tile_sched_params)
        # LOG2_E = math.log2(math.e) # I think we can multiply by 1/sqrt(d) here too

        # we should be ready to do the kernel now
        print("launching kernel")
        self.kernel(tma_tensor_q, tma_tensor_k, tma_tensor_v, tma_tensor_o, tma_atom_q, self.sQ_layout, SingleTileScheduler, tile_sched_params).launch(grid=grid_dim, block=[self.num_threads, 1, 1], stream=stream)
    
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, tma_atom_q: cute.CopyAtom, sQ_layout: cute.ComposedLayout, TileScheduler: cutlass.Constexpr[Callable], tile_sched_params: ParamsBase):
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
            tx_count=0, # should add tma copy bytes [k]
            defer_sync=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # need to know how many blocks to iterate through here

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        print0("at branching")
        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(mQ, sQ, tma_atom_q, pipeline_k, TileSchedulerCls)
        else:
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            self.mma(pipeline_k)
    
    @cute.jit
    def load(self, mQ: cute.Tensor, sQ: cute.Tensor, tma_atom_q: cute.CopyAtom, pipeline_k: pipeline.PipelineAsync, TileSchedulerCls: Callable):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            m_block, head_idx, batch_idx = work_tile.tile_idx
            mQ_curr = mQ[None, None, head_idx, batch_idx]

            gQ = cute.local_tile(mQ_curr, (self.tile_m, self.hdimk), (m_block, 0))
            load_Q, _, _ = my_utils.tma_get_copy_fn(
                tma_atom_q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )

            # 1824: First pipeline K, just add Q in there
            pipeline_k.producer_acquire(
                kv_producer_state,
                extra_tx_count=self.tma_copy_bytes["Q"]
            )
            load_Q(tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))
    
    @cute.jit
    def mma(self, pipeline_k: pipeline.PipelineAsync):
        kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
        # pipeline_k.consumer_wait(kv_consumer_state)

        
        
        
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
        gckv = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() # no multicast for now
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
            1
        )
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            gckv,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.hdimv),
            1
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            gso,
            mO,
            self.sO_layout,
            (self.tile_m, self.hdimv)
        )
        return tma_atom_q, tma_tensor_q, tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v, tma_atom_o, tma_tensor_o


if __name__ == "__main__":
    q = torch.randn((16, 16, 1024, 64), dtype=torch.bfloat16).to('cuda')
    k = torch.randn((16, 16, 1024, 64), dtype=torch.bfloat16).to('cuda')
    v = torch.randn((16, 16, 1024, 64), dtype=torch.bfloat16).to('cuda')
    o = torch.empty((16, 16, 1024, 64), dtype=torch.bfloat16).to('cuda')

    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
    )
    [q_cute, k_cute, v_cute, o_cute] = [convert_from_dlpack(x) for x in (q, k, v, o)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    fa = FlashSM90(dtype=cutlass.BFloat16, qk_mn=(128, 256), cluster_size_m=1)
    fa(q_cute, k_cute, v_cute, o_cute, current_stream)