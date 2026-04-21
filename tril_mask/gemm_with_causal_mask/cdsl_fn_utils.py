import torch
import functools
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from typing import Optional

import cutlass.cute as cute

STREAM = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

def jit_cache(fn):
    cache = dict()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = args + tuple(sorted(kwargs.items())) if kwargs else args
        if key in cache:
            return cache[key]
        
        compiled_fn = fn(*args, **kwargs)
        cache[key] = compiled_fn
        return compiled_fn
    
    wrapper.cache = cache
    return wrapper


# convert_from_dlpack = lambda tensor: (
#         from_dlpack(tensor.detach(), assumed_align=16)
#     )

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
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