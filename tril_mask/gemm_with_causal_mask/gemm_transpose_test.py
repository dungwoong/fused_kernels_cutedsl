from typing import Tuple

import torch
import cutlass
from cutlass import cute, pipeline

from my_runtime import shared, mma, pipeline as my_pipeline
from cdsl_fn_utils import compile_cutedsl_no_stream
import my_utils

"""
A: 16x128
B: 128x128

compute (BtAt)t = m128n16k128 --> (16,128)
"""

EPI_BAR = 1


def get_epi_tensor_atom(t: cute.Tensor, epi_smem_layout_staged: cute.ComposedLayout, epi_tile: Tuple[int, int]):
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    epi_tma_tensor_layout = cute.composition(cute.make_identity_layout(t.shape), epi_tile)
    op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op, t, epi_smem_layout, epi_tma_tensor_layout
    )
    return tma_atom, tma_tensor


class Kernel:
    def __init__(self):
        self.tile_m, self.tile_n, self.tile_k = (128, 16, 64)
        self.stages = 2
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        self.consumer_wgs = None
        self.consumer_warps = None
        self.consumer_regs = 232
        self.producer_regs = 40
    
    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
        # BtAt
        As_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, self.tile_n, self.tile_k, self.stages)
        Bs_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, self.tile_m, self.tile_k, self.stages)
        epi_layout = shared.get_smem_layout_row_major(cutlass.BFloat16, self.tile_m, self.tile_n, 1)

        # Let's just say there's only one block to work on
        tiled_gemm = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.tile_m, self.tile_n)
        self.consumer_wgs = tiled_gemm.size // 128
        self.consumer_warps = self.consumer_wgs * 4
        
        A_g2s_atom, A_g2s_tensor = shared.get_tma_tensor_and_atom(A, As_layout, self.tile_n, self.tile_k)
        B_g2s_atom, B_g2s_tensor = shared.get_tma_tensor_and_atom(B, Bs_layout, self.tile_m, self.tile_k)
        C_s2g_atom, C_s2g_tensor = get_epi_tensor_atom(C, epi_layout, (self.tile_m, self.tile_n))

        self.kernel(A_g2s_atom, A_g2s_tensor, B_g2s_atom, B_g2s_tensor, C_s2g_atom, C_s2g_tensor, epi_layout, tiled_gemm, As_layout, Bs_layout).launch(grid=[1, 1, 1], block=[(self.consumer_wgs + 1) * 128])
    
    @cute.kernel
    def kernel(self, A_g2s_atom: cute.CopyAtom, A_g2s_tensor: cute.Tensor, B_g2s_atom: cute.CopyAtom, B_g2s_tensor: cute.Tensor, C_s2g_atom: cute.CopyAtom, C_s2g_tensor: cute.Tensor, epi_layout: cute.ComposedLayout, tiled_gemm: cute.TiledMma, As_layout: cute.ComposedLayout, Bs_layout: cute.ComposedLayout):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        
        SharedStorage = type("SharedStorage", (), dict())
        SharedStorage.__annotations__['As_ptr'] = shared.memrange(self.dtype, As_layout, 1024)
        SharedStorage.__annotations__['Bs_ptr'] = shared.memrange(self.dtype, Bs_layout, 1024)
        SharedStorage.__annotations__['epi_ptr'] = shared.memrange(self.dtype, epi_layout, 1024)
        SharedStorage.__annotations__['pipe_ptr'] = cute.struct.MemRange[cutlass.Int64, self.stages * 2]
        s_alloc = cutlass.utils.SmemAllocator()
        smem = s_alloc.allocate(cute.struct(SharedStorage))

        As = shared.smem_get_tensor(smem, 'As_ptr', As_layout)
        Bs = shared.smem_get_tensor(smem, 'Bs_ptr', Bs_layout)
        Cs = shared.smem_get_tensor(smem, 'epi_ptr', epi_layout)

        n_bytes = (
            cute.size_in_bytes(cutlass.BFloat16, cute.select(As_layout, mode=[0, 1])) + 
            cute.size_in_bytes(cutlass.BFloat16, cute.select(Bs_layout, mode=[0, 1])))
        pipe = my_pipeline.make_tma_pipeline(
            smem.pipe_ptr.data_ptr(),
            self.stages,
            self.consumer_warps,
            num_bytes=n_bytes
        )

        sliced_A = cute.local_tile(A_g2s_tensor, (self.tile_n, self.tile_k), (0, None))
        k_iters = cute.size(sliced_A, mode=[2])

        if (warp_idx < self.consumer_warps): # CONSUMER
            cute.arch.setmaxregister_increase(self.consumer_regs)
            cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            acc = mma.get_acc(tiled_gemm, self.tile_m, self.tile_n, self.acc_dtype)
            gemm_accumulate = False
            for k2 in cutlass.range(k_iters, unroll=1):
                pipe.consumer_wait(cstate)
                mma.accumulating_gemm_ss(tidx, tiled_gemm, Bs, As, acc, cstate, cstate, gemm_accumulate, 0)
                gemm_accumulate = True
                pipe.consumer_release(cstate)
            

            rO = cute.make_fragment_like(acc, self.dtype)
            rO.store(acc.load().to(self.dtype))

            Cs_slice = Cs[None, None, 0]
            copy_atom_C = my_utils.get_smem_store_atom(90, self.dtype, False)
            thr_copy_r2s = cute.make_tiled_copy_C(copy_atom_C, tiled_gemm).get_slice(tidx)
            # for now, let's say we're computing AtBt
            r2s_s = thr_copy_r2s.partition_D(Cs_slice)
            r2s_r = thr_copy_r2s.retile(rO)
            cute.copy(copy_atom_C, r2s_r, r2s_s)
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            cute.arch.barrier_arrive(barrier_id=EPI_BAR, number_of_threads=(self.consumer_warps*32) + 32)

            gO = cute.local_tile(C_s2g_tensor, (self.tile_m, self.tile_n), (bidy, bidx))
            store_O, _, _ = my_utils.tma_get_copy_fn(
                C_s2g_atom, 0, cute.make_layout(1), Cs_slice, gO, single_stage=True
            )
            if warp_idx == 0:
                cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=(self.consumer_warps*32) + 32)
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

        if (warp_idx >= self.consumer_warps): # PRODUCER
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if (warp_idx == self.consumer_warps):
                pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                for k1 in cutlass.range(k_iters, unroll=1):
                    pipe.producer_acquire(pstate)
                    shared.tma_copy(A_g2s_atom, A_g2s_tensor, As, self.tile_n, self.tile_k, bidx, k1, pipe, pstate)
                    shared.tma_copy(B_g2s_atom, B_g2s_tensor, Bs, self.tile_m, self.tile_k, bidy, k1, pipe, pstate)


if __name__ == '__main__':
    print('staerting...')
    # Compute (BtAt)t
    m, n, k = 16, 128, 128
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')

    # should be m, n but ok
    c = torch.zeros((n, m), dtype=torch.bfloat16).to('cuda')

    kernel = Kernel()
    compiled_kernel = compile_cutedsl_no_stream((a, b, c), kernel)
    compiled_kernel(a, b, c)