from typing import Type
import cutlass
from cutlass import cute
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils

# this is the library we'd import, generated code can use any functions from this lib


# TODO experiment to see how column-major works
def get_smem_layout_row_major(
    dtype: Type[cutlass.Numeric],
    rows: int,
    cols: int,
    stages: int,
):
    atom = cute.nvgpu.warpgroup.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, cols), dtype)
    layout = cute.tile_to_shape(atom, (rows, cols, stages), (0, 1, 2))
    return layout


def get_smem_struct(fields):
    """
    dict is name: type
    """
    cls = type("SharedStorage", (), dict())
    cls.__annotations__ = fields
    return cute.struct(cls)  # maybe we can cute.struct later


def smem_get_tensor(storage, field_name, layout):
    # sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    return getattr(storage, field_name).get_tensor(layout.outer, swizzle=layout.inner)


def memrange(dtype, smem_layout, align):
    return cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout)], align]


def get_tma_tensor_and_atom(tG, shared_layout, rows, cols):
    return cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        tG,
        cute.select(shared_layout, mode=[0, 1]),
        (rows, cols),
    )


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    **kwargs,
):
    """Returns a callable to perform the G2S copy"""
    src_is_smem = cutlass.const_expr(isinstance(src_tensor.iterator, cute.Pointer) and src_tensor.memspace == cute.AddressSpace.smem)
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)

    s, g = cute.nvgpu.cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - 1),
        cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - 2),
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    # TODO might need to fix
    def copy_tma(src_row, src_col, dst_idx, **kwargs2):
        cute.copy(atom, src[None, src_row, src_col], dst[None, dst_idx], **kwargs2, **kwargs)

    return copy_tma, s, g


# @cute.jit
def tma_copy(
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    s_tensor: cute.Tensor,
    tile_m: int,
    tile_n: int,
    src_row: int,
    src_col: int,
    pipe: cutlass.pipeline.PipelineAsync,
    state: cutlass.pipeline.PipelineState,
    cta_coord: cute.Coord=0,
    cta_layout: cute.Layout=None,
    mcast_mask: any=0,
):
    if cta_layout is None:
        cta_layout = cute.make_layout((1, 1))
    gT = cute.local_tile(tma_tensor, (tile_m, tile_n), (None, None))
    load, _, _ = tma_get_copy_fn(
        tma_atom,
        cta_coord,
        cta_layout,
        gT,
        s_tensor,
    )
    # return load
    load(src_row, src_col, state.index, tma_bar_ptr=pipe.producer_get_barrier(state), mcast_mask=mcast_mask)

