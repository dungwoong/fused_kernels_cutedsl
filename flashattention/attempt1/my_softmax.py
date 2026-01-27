from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, override
from functools import lru_cache
import operator
import math

import cutlass
from cute_dsl_utils import ArgumentsBase, ParamsBase
from fast_math import FastDivmod
from cutlass import Int32, Boolean
from cutlass import cute, pipeline
from cutlass._mlir import ir
import my_utils


@dataclass
class Softmax(ParamsBase):
    scale_log2: cutlass.Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    # softmax_scale: cutlass.Float32 | None = None

    @staticmethod
    def create(scale_log2: cutlass.Float32, num_rows: cutlass.Constexpr[int]):
        row_max = cute.make_rmem_tensor(num_rows, cutlass.Float32)
        row_sum = cute.make_rmem_tensor(num_rows, cutlass.Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum)
    
    def reset(self) -> None:
        self.row_max.fill(-cutlass.Float32.inf)
        self.row_sum.fill(0.0)
    
    def _compute_row_max() -> cutlass.Float32:
        pass

    def _compute_row_sum() -> cutlass.Float32:
        pass

    @cute.jit
    def online_softmax(self, acc_S: cute.Tensor, is_first: cutlass.Constexpr[bool]=False, check_inf: cutlass.Constexpr[bool] = True):
        """
        This function just accumulates the sums and returning a row scale that we can use to scale the P tile
        The scale is exp(max_prev - max_cur)
        """
        acc_S_mn = my_utils.make_acc_tensor_mn_view(acc_S) # new modes are (rows, cols)
        row_scale = cute.make_fragment_like(self.row_max, cutlass.Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load() # load the row
            
            # rowmax
            row_max_cur = my_utils.fmax_reduce(acc_S_row, init_val=row_max[r] if cutlass.const_expr(not is_first) else None)
            row_max_cur = my_utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)

            # rowmax prev/cur updated
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -cutlass.Float32.inf else row_max_cur
            
            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = my_utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled) # exp2f(scale_log2(s - max)) = exp(s-max)
                acc_S_row_sum = my_utils.fadd_reduce(acc_S_row_exp, init_val=None) # sum(exp(s-max))
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = my_utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled) # exp(s-max)
                row_scale[r] = my_utils.exp2f((row_max_prev - row_max_cur) * scale_log2) # rowscale = exp(mprev - mcur)
                acc_S_row_sum = my_utils.fadd_reduce(acc_S_row_exp, init_val=row_sum[r] * row_scale[r]) # rowsum = sum(exp(x-max)) + prevsum * rowscale

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale
    
    @cute.jit
    def finalize(self) -> cute.Tensor:
        """
        Calculates final scale = 1/rowsum
        """
        # they have sink_val and save log-sum-exp and stuff, but I don't need that
        
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        row_sum.store(my_utils.warp_reduce(row_sum.load(), operator.add, width=4)) # accumulate sum across the warp
        row_scale = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            # if cutlass.const_expr(sink_val is not None):
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            )
        return row_scale
        
    
    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_O_mn = my_utils.make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])