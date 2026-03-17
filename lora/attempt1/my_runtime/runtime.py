from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import torch


def convert_from_dlpack(tensor):
    return from_dlpack(tensor.detach(), assumed_align=16)


def to_cute_tensors(*tensors):
    return [convert_from_dlpack(t) for t in tensors]


def get_current_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
