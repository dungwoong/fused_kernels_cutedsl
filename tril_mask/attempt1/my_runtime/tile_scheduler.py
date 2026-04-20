from typing import Tuple
from cck.runtime.utils.cute_dsl_utils import ParamsBase, ArgumentsBase
from cck.runtime.utils.fast_math import FastDivmod
from dataclasses import dataclass
import math

import cutlass
from cutlass import cute
from cutlass import Int32, const_expr

MAX_SM_H100 = 132


@dataclass
class Gemm2DTileSchedulerArguments(ArgumentsBase):
    ntiles_mnl: cute.Shape
    is_persistent: cutlass.Constexpr[bool]

    @staticmethod
    @cute.jit
    def create(t: cute.Tensor, tile_m: int, tile_n: int, is_persistent: bool):
        assert cute.rank(t) <= 3  # either mn or mnl
        shape = t.shape if cute.rank(t.shape) == 3 else (*t.shape, 1)
        ntiles_mnl = cute.ceil_div(shape, (tile_m, tile_n, 1))
        return Gemm2DTileSchedulerArguments(ntiles_mnl, is_persistent)


class Gemm2DTileScheduler:
    """
    Assumes that the matrix layout is (r, c, b)
    We just iterate down (r, c) and then b.

    This scheduler performs row-first and iterates down
    the batch dimension 1 at a time.

    This scheduler is quite inefficient due to caching,
    it would be best to form groups
    """

    @dataclass
    class Params(ParamsBase):
        n_ctas_mnl: cute.Shape
        n_ctas_mn: cute.Shape
        is_persistent: cutlass.Constexpr[bool]
        n_ctas_mn_divmod: FastDivmod
        n_ctas_n_divmod: FastDivmod

        @staticmethod
        @cute.jit
        def create(args: Gemm2DTileSchedulerArguments, *, loc=None, ip=None):
            mnl = args.ntiles_mnl
            mn = cute.select(args.ntiles_mnl, [0, 1])
            mn_prod = math.prod(mn)
            n = args.ntiles_mnl[1]
            return Gemm2DTileScheduler.Params(
                n_ctas_mnl=mnl,
                n_ctas_mn=mn,
                is_persistent=args.is_persistent,
                n_ctas_mn_divmod=FastDivmod.create(mn_prod),
                n_ctas_n_divmod=FastDivmod.create(n),
            )

    def __init__(self, current_work_idx: Int32, num_tiles_executed: Int32, params: Params, *, loc=None, ip=None):
        self._current_work_idx = current_work_idx
        self._num_tiles_executed = num_tiles_executed
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: Gemm2DTileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return Gemm2DTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "Gemm2DTileScheduler":
        # both rasterization schemes put all blocks in bidx
        bidx, _, _ = cute.arch.block_idx()
        return Gemm2DTileScheduler(Int32(bidx), Int32(0), params, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: Params, max_sms: Int32, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        if const_expr(not params.is_persistent):
            return (math.prod(params.n_ctas_mnl), 1, 1)
        else:
            return (max_sms, 1, 1)

    @cute.jit
    def _map_cta_coords(self, work_id: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32]:
        params = self.params
        batch, work_id_in_problem = params.n_ctas_mn_divmod.divmod(work_id)
        row, col = params.n_ctas_n_divmod.divmod(work_id_in_problem)
        return row, col, batch

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None):
        params = self.params
        r, c, b = self._map_cta_coords(self._current_work_idx)
        tile_coord_mnkl = (r, c, None, b)
        if const_expr(not params.is_persistent):
            is_valid = self._num_tiles_executed == 0
        else:
            is_valid = self._current_work_idx < cute.size(params.n_ctas_mnl)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def fetch_next_work(self, *, loc=None, ip=None):
        pass

    @cute.jit
    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.is_persistent):
            num_persistent_ctas = cute.arch.grid_dim()[0]
            self._current_work_idx += Int32(num_persistent_ctas)

        self._num_tiles_executed += Int32(1)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self._num_tiles_executed,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self._num_tiles_executed,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


def n_warps(*args, warp_specialize=False):
    """Assumes simple thread setup"""
    nthreads = sum(a.size for a in args)
    assert nthreads % 32 == 0
    return (nthreads // 32) + (4 if warp_specialize else 0)
