import torch
import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import utils, pipeline
from cdsl_helpers import shared
import my_utils
from cdsl_fn_utils import compile_cutedsl

DTYPE = cutlass.Float32
DTYPE_TORCH = torch.float32
BUF_ALIGN = 1024

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

class Kernel:
    def __init__(self, tile_m, tile_n):
        self.tile_m, self.tile_n = tile_m, tile_n
        self.smem_layout = None
        self.smem_cls = None
        self.n_bytes = 0
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream):
        self._set_smem_layouts()
        self._set_shared_storage()
        self.n_bytes = cute.size_in_bytes(DTYPE, self.smem_layout)
        atom_ld, tensor_ld, atom_st, tensor_st = self._get_tma_copy_attrs(mA, mO)
        self.kernel(
            atom_ld, tensor_ld, atom_st, tensor_st, self.smem_layout
        ).launch(grid=[1024], block=[128, 1, 1])
    
    @cute.kernel
    def kernel(self, atom_ld, tensor_ld, atom_st, tensor_st, smem_layout):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.smem_cls)

        pipeline_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1)
        pipeline_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, size=4)
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline_prod_grp,
            consumer_group=pipeline_cons_grp,
            tx_count=self.n_bytes,
            defer_sync=False,
        )

        sO = storage.sO.get_tensor(smem_layout.outer, swizzle=smem_layout.inner)
        prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        if warp_idx == 0:
            pipe.producer_acquire(prod_state)
            shared.tma_copy(atom_ld, tensor_ld, sO, self.tile_m, self.tile_n, 0, 0, pipe, prod_state)
        pipe.consumer_wait(cons_state)

        cute.arch.sync_threads()

        # honestly everything is happening in the async proxy so I think I can just go for the store
        store_O, _, _ = my_utils.tma_get_copy_fn(
            atom_st, 0, cute.make_layout(1), sO, tensor_st, single_stage=True
        )
        if warp_idx == 0:
            store_O()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)

    def _set_smem_layouts(self):
        smem_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, DTYPE, self.tile_n),
            DTYPE
        )
        self.smem_layout = cute.tile_to_shape(
            smem_atom, (self.tile_m, self.tile_n, 1), (0, 1, 2)
        )
    
    def _set_shared_storage(self):
        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, 2]
            sO: cute.struct.Align[cute.struct.MemRange[DTYPE, cute.cosize(self.smem_layout)], BUF_ALIGN]
        
        self.smem_cls = SharedStorage

    def _get_tma_copy_attrs(self, mA: cute.Tensor, mO: cute.Tensor):
        atom_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        atom_store = cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(cute.nvgpu.cpasync.ReductionOp.ADD)

        tma_atom_load, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            atom_load,
            mA,
            cute.select(self.smem_layout, mode=[0, 1]),
            (self.tile_m, self.tile_n)
        )

        tma_atom_store, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            atom_store,
            mO,
            cute.select(self.smem_layout, mode=[0, 1]),
            (self.tile_m, self.tile_n)
        )

        return tma_atom_load, tma_tensor_a, tma_atom_store, tma_tensor_o

if __name__ == '__main__':
    k = Kernel(128, 128)
    a = torch.ones((128, 128), dtype=DTYPE_TORCH, device='cuda')
    a[:, 1] = a[:, 1] * 2 # check indexing is right by setting a column differently
    o = torch.zeros_like(a)
    # print(o)

    compiled_kernel = compile_cutedsl((a, o), k)
    compiled_kernel(a, o, None)
    compiled_kernel(a, o, None) # should give us 2048
    print(o)