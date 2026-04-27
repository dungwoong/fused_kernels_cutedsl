import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import shared
import math

"""
This didn't really go anywhere but I'll just keep it here anyways
"""

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )


TILE_M, TILE_N = 16, 128

class Kernel:
    def __init__(self):
        self.tile_m = TILE_M
        self.tile_n = TILE_N
        self.dtype = cutlass.BFloat16
        self.num_threads = 32
    
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor):
        # smem_layout = shared.get_smem_layout_row_major(self.dtype, self.tile_m, self.tile_n, 1)
        smem_layout = self._make_smem_layout()
        copy_a = self._make_copy(self.tile_m, self.tile_n)
        self.kernel(a, b, smem_layout, copy_a).launch(grid=[1], block=[self.num_threads])
    
    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, smem_layout: cute.ComposedLayout, copy_a: cute.TiledCopy):
        """
        Workload: take a and copy to b.
        """
        tidx, _, _ = cute.arch.thread_idx()

        SharedStorage = type("SharedStorage", (), dict())
        SharedStorage.__annotations__['As_ptr'] = shared.memrange(self.dtype, smem_layout, 1024)
        SharedStorage.__annotations__['Bs_ptr'] = shared.memrange(self.dtype, smem_layout, 1024)
        smem_allocator_ = cutlass.utils.SmemAllocator()
        smem__ = smem_allocator_.allocate(cute.struct(SharedStorage))
        As = shared.smem_get_tensor(smem__, 'As_ptr', smem_layout)
        Bs = shared.smem_get_tensor(smem__, 'Bs_ptr', smem_layout)

        a_tiled = cute.local_tile(a, (self.tile_m, self.tile_n), (0, 0))
        b_tiled = cute.local_tile(b, (self.tile_m, self.tile_n), (0, 0))

        thr_copy = copy_a.get_slice(tidx)
        tAgA = thr_copy.partition_S(a_tiled)
        tAsA = thr_copy.partition_D(As)
        # print(As)
        # print(copy_a)
        # print(a_tiled)
        cute.copy(copy_a, tAgA, tAsA)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # make the mma op

        cute.autovec_copy(As, b_tiled)
    
    def _make_copy(self, rows, cols, copy_bits=128):
        atom_copy = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits)
        copy_elems = copy_bits // self.dtype.width
        shape_dim_1 = cols // copy_elems # no. copies along N
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)
    
    # not sure why but 128 byte loads aren't working here, you get misaligned address
    # swizzle(3, 3, 3) is not working
    def _make_smem_layout(self):
        # print(int(math.log2(self.tile_n * self.dtype.width // copy_bits)))
        swizzle_bits = 4
        layout_atom_outer = (
            cute.make_layout((8, self.tile_n), stride=(self.tile_n, 1))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3), # NOTE second number must be 4, I think second number is MBase actually
            0,
            layout_atom_outer
        )
        layout = cute.tile_to_shape(layout_atom, (self.tile_m, self.tile_n), (0, 1))
        return layout

if __name__ == '__main__':
    a = torch.randn((TILE_M, TILE_N), dtype=torch.bfloat16, device='cuda')
    b = torch.empty_like(a)
    a_cute, b_cute = convert_from_dlpack(a), convert_from_dlpack(b)
    k = Kernel()
    compiled_kernel = cute.compile(k, a_cute, b_cute, options='--enable-tvm-ffi')
    compiled_kernel(a, b)
    print(b - a)