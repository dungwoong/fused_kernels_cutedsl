import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from typing import Optional

import cutlass.cute as cute

"""
Utility functions for compiling cuteDSL kernels.

Currently, two options are:
- pass in sample tensors using convert_from_dlpack with sample inputs
- create faketensors
compile_cutedsl uses the first strategy
"""

STREAM = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
    )

def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )

def compile_cutedsl(tensors, kernel):
    cute_tensors = [convert_from_dlpack(t) for t in tensors]
    compiled = cute.compile(kernel, *cute_tensors, STREAM, options='--enable-tvm-ffi')
    return compiled

def compile_cutedsl_no_stream(tensors, kernel):
    cute_tensors = [convert_from_dlpack(t) for t in tensors]
    compiled = cute.compile(kernel, *cute_tensors, options='--enable-tvm-ffi')
    return compiled