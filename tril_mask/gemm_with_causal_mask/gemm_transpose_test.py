import torch
import cutlass
from cutlass import cute, pipeline

from my_runtime import shared, mma, pipeline as my_pipeline
from cdsl_fn_utils import compile_cutedsl_no_stream

"""
A: 16x128
B: 128x128

compute (BtAt)t = m128n16k128 --> (16,128)
"""

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

        # Let's just say there's only one block to work on
        tiled_gemm = mma.get_tiled_mma(self.dtype, True, True, self.acc_dtype, self.tile_m, self.tile_n)
        self.consumer_wgs = tiled_gemm.size // 128
        self.consumer_warps = self.consumer_wgs * 4
        
        A_g2s_atom, A_g2s_tensor = shared.get_tma_tensor_and_atom(A, As_layout, self.tile_n, self.tile_k)
        B_g2s_atom, B_g2s_tensor = shared.get_tma_tensor_and_atom(B, Bs_layout, self.tile_m, self.tile_k)

        self.kernel(A_g2s_atom, A_g2s_tensor, B_g2s_atom, B_g2s_tensor, tiled_gemm, As_layout, Bs_layout).launch(grid=[1, 1, 1], block=[(self.consumer_wgs + 1) * 128])
    
    @cute.kernel
    def kernel(self, A_g2s_atom: cute.TiledCopy, A_g2s_tensor: cute.Tensor, B_g2s_atom: cute.TiledCopy, B_g2s_tensor: cute.Tensor, tiled_gemm: cute.TiledMma, As_layout: cute.ComposedLayout, Bs_layout: cute.ComposedLayout):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        
        SharedStorage = type("SharedStorage", (), dict())
        SharedStorage.__annotations__['As_ptr'] = shared.memrange(self.dtype, As_layout, 1024)
        SharedStorage.__annotations__['Bs_ptr'] = shared.memrange(self.dtype, Bs_layout, 1024)
        SharedStorage.__annotations__['pipe_ptr'] = cute.struct.MemRange[cutlass.Int64, self.stages * 2]
        s_alloc = cutlass.utils.SmemAllocator()
        smem = s_alloc.allocate(cute.struct(SharedStorage))

        As = shared.smem_get_tensor(smem, 'As_ptr', As_layout)
        Bs = shared.smem_get_tensor(smem, 'Bs_ptr', Bs_layout)

        n_bytes = (
            cute.size_in_bytes(cutlass.BFloat16, cute.select(As_layout, mode=[0, 1])) + 
            cute.size_in_bytes(cutlass.BFloat16, cute.select(Bs_layout, mode=[0, 1])))
        pipe = my_pipeline.make_tma_pipeline(
            smem.pipe_ptr.data_ptr(),
            self.stages,
            self.consumer_warps,
            num_bytes=n_bytes
        )

        sliced_A = cute.local_tile(A_g2s_atom, (self.tile_n, self.tile_k), (0, None))
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

        if (warp_idx >= self.consumer_warps): # PRODUCER
            cute.arch.setmaxregister_decrease(self.producer_regs)
            if (warp_idx == self.consumer_warps):
                pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                for k1 in cutlass.range(k_iters, unroll=1):
                    pipe.producer_acquire(pstate)
                    shared.tma_copy(A_g2s_atom, A_g2s_tensor, As, self.tile_n, self.tile_k, bidx, k1, pipe, pstate)
                    shared.tma_copy(B_g2s_atom, B_g2s_tensor, Bs, self.tile_m, self.tile_k, bidy, k1, pipe, pstate)


if __name__ == '__main__':
    # Compute (BtAt)t
    m, n, k = 16, 128, 128
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')

    # should be m, n but ok
    c = torch.zeros((n, m), dtype=torch.bfloat16).to('cuda')

    kernel = Kernel()
    compiled_kernel = compile_cutedsl_no_stream((a, b, c), kernel)
    compiled_kernel(a, b, c)