import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import shared
import math

"""
- 64x16 region
- load to smem (64, 16)
- ldmatrix (16, 16) per WG
- stmatrix (16, 64)
- store (16, 64)

This kernel:
- swaps the input memory to (64, 16)
"""
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

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )

def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


TILE_M, TILE_N = 64, 16

class Kernel:
    def __init__(self):
        self.tile_m = TILE_M
        self.tile_n = TILE_N
        self.dtype = cutlass.BFloat16
        self.num_threads = 128
    
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor):
        # smem_layout = shared.get_smem_layout_row_major(self.dtype, self.tile_m, self.tile_n, 1)
        smem_layout = self._make_smem_layout(self.tile_m, self.tile_n, 4) # (16, 64)
        smem_layout_2 = self._make_smem_layout(self.tile_n, self.tile_m, 0) # (64, 16)
        copy_a = self._make_copy(self.tile_m, self.tile_n)
        copy_b = self._make_copy(self.tile_n, self.tile_m)
        self.kernel(a, b, smem_layout, smem_layout_2, copy_a, copy_b).launch(grid=[1], block=[self.num_threads])
    
    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, smem_layout: cute.ComposedLayout, smem_layout_b: cute.ComposedLayout, copy_a: cute.TiledCopy, copy_b: cute.TiledCopy):
        """
        Workload: take a and copy to b.
        """
        tidx, _, _ = cute.arch.thread_idx()

        SharedStorage = type("SharedStorage", (), dict())
        SharedStorage.__annotations__['As_ptr'] = shared.memrange(self.dtype, smem_layout, 1024)
        SharedStorage.__annotations__['Bs_ptr'] = shared.memrange(self.dtype, smem_layout_b, 1024)
        smem_allocator_ = cutlass.utils.SmemAllocator()
        smem__ = smem_allocator_.allocate(cute.struct(SharedStorage))
        As = shared.smem_get_tensor(smem__, 'As_ptr', smem_layout)
        Bs = shared.smem_get_tensor(smem__, 'Bs_ptr', smem_layout_b)
        # At_ptr = cute.recast_ptr(As.iterator, smem_layout_b.inner, dtype=self.dtype)
        # As_t = cute.make_tensor(At_ptr, smem_layout_b.outer)

        a_tiled = cute.local_tile(a, (self.tile_m, self.tile_n), (0, 0))
        b_tiled = cute.local_tile(b, (self.tile_n, self.tile_m), (0, 0))

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

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.dtype, cutlass.Float32, (16, 8, 16)
        )

        perm_n = cute.make_layout((8, 2, 2), stride=(1, 16, 8))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            (4, 1, 1), # this controls how many threads participate
            # the permutation is there so it will permute the input region before tiling
            # otherwise e.g. say you want 2 mmas like W0: 01 W1: 23, it would instead do 01 then 23 so W0:02 W1:13
        )
        # ok so this will do 64x8, but you need to partition a shape to apply it
        partition_a = tiled_mma.partition_shape_A((64, 16))
        print('partition_a:', partition_a) # ((2, 2, 2), 1, 1)
        a_regs = cute.make_rmem_tensor(partition_a, self.dtype)

        # we want 0-7, 16-23, 32-... | 8-15, 24-31, etc.
        # assign every 8 elements to warps 1 2 3 4 etc.
        perm_n = cute.make_layout((8, 4, 2), stride=(1, 16, 8))
        tiled_mma_b = cute.make_tiled_mma(
            mma_op,
            (1, 4, 1),
            permutation_mnk=(16, perm_n, 16)
        )
        partition_b = tiled_mma_b.partition_shape_B((16, 64))
        print('TILED_MMA_B:', tiled_mma_b)
        print('partition_b:', partition_b) # (2, 2), 1, 4 since each packed item has 2 things

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
            self.dtype
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(True, 4),
            self.dtype
        )
        composed = cute.composition(
            cute.make_layout((4, 1), stride=(1, 0)),
            copy_atom.layout_dst_tv,
        )
        # tiled_copy = cute.make_tiled_copy(copy_atom, composed, (16, 16))
        # HACK don't deal with layouts it's so stupid...
        tiled_copy = cute.make_tiled_copy_A(copy_atom, tiled_mma)
        thr_copy = tiled_copy.get_slice(tidx)
        a_shared_retiled = thr_copy.partition_S(As)
        a_regs_retiled = thr_copy.retile(a_regs)
        print('a_regs_retiled:', a_regs_retiled)
        print('a_shared_retiled:', a_shared_retiled)
        print('TILED_COPY:', tiled_copy)
        
        # this expects 64, 16
        Bs_t = transpose_view(Bs)
        # tiled_store = cute.make_tiled_copy_B(store_atom, tiled_mma_b)
        # thr_store = tiled_store.get_slice(tidx)
        # b_shared_retiled = thr_store.partition_D(As)
        # # b_regs_retiled = thr_store.retile(a_regs)
        # b_regs_retiled = thr_store.retile(a_regs)
        # print('Bs:', Bs)
        # print('Bst:', Bs_t)

        # Try storing back to As where they came from as a prelim: did not work
        # stmatrix is SM_90 or higher, so try this on a hopper GPU, it should work.
        """
        store_atom = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4),
            self.dtype
        )
        tiled_store = cute.make_tiled_copy_A(store_atom, tiled_mma)
        thr_store = tiled_store.get_slice(tidx)
        b_shared_retiled = thr_store.partition_D(As)
        b_regs_retiled = thr_store.retile(a_regs)
        """


        # print('TILED_STORE:', tiled_store)
        # print('b_regs_retiled:', b_regs_retiled)
        # print('b_shared_retiled:', b_shared_retiled)


        # print(tiled_copy)
        # print(tiled_store)

        # cute.autovec_copy(As, Bs_t)
        # if tidx == 0:
        #     for i in cutlass.range(cute.size(As)):
        #         Bs_t[i] = As[i]
        cute.copy(tiled_copy, a_shared_retiled[None, None, 0], a_regs_retiled[None, None, 0])
        # cute.copy(tiled_store, a_regs_retiled[None, None, 0], a_shared_retiled[None, None, 0])
        # print0(b_regs_retiled)
        thr_mma = tiled_mma.get_slice(tidx)
        partitioned_Bs = thr_mma.partition_A(Bs_t)
        print(partitioned_Bs)
        cute.autovec_copy(a_regs_retiled, partitioned_Bs)

        cute.arch.sync_threads()

        # cute.autovec_copy(As, b_tiled)
        cute.autovec_copy(Bs, b_tiled)
    
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
    def _make_smem_layout(self, m, n, swizzle_bits=4):
        # print(int(math.log2(self.tile_n * self.dtype.width // copy_bits)))
        layout_atom_outer = (
            cute.make_layout((8, n), stride=(n, 1))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3), # NOTE second number must be 4, I think second number is MBase actually
            0,
            layout_atom_outer
        )
        layout = cute.tile_to_shape(layout_atom, (m, n), (0, 1))
        return layout

if __name__ == '__main__':
    # a = torch.ones((TILE_M, TILE_N), dtype=torch.bfloat16, device='cuda')
    lst = [ i for i in range(16*16)]
    lst = lst * 4
    a = torch.tensor(lst, dtype=torch.bfloat16, device='cuda').reshape(TILE_M, TILE_N)
    b = torch.zeros((TILE_N, TILE_M), dtype=torch.bfloat16, device='cuda')
    a_cute, b_cute = convert_from_dlpack(a), convert_from_dlpack(b)
    k = Kernel()
    compiled_kernel = cute.compile(k, a_cute, b_cute, options='--enable-tvm-ffi')
    compiled_kernel(a, b)
    print(f'went from {a.shape} --> {b.shape}')
    print(b - a.t())
    # print(b)
    # print(a)