import argparse
from typing import Callable, Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton import runtime
from triton.testing import do_bench
import functools
import statistics
import multiprocessing as mp

from cutlass import cute
from cutlass.cute.runtime import from_dlpack

import pytest
import io
import sys
import traceback

from attn import FlashSM90

torch.manual_seed(42)
OUTPUT_FILE = open("timings.csv", "w")
OUTPUT_FILE.write("bs,nh,seqlen,dim,latency_ms,tflops,method\n")

# I don't have pytest on my sif so I'll try something manual

def get_tflops(bs, nh, lq, lkv, head_dim, head_dim_v, latency_ms):
    qk = bs * nh * (2 * lq * lkv * head_dim)
    smx = bs * nh * (4 * lq * lkv) # max + sub + exp + sum + div, but codebases(e.g. thunderkittens) use 4
    kv = bs * nh * (2 * lq * lkv * head_dim_v)
    return (qk + smx + kv) / latency_ms / 1e9

# if output has a mean of 0, we get a large relative error
def generate_input(*shape):
    return torch.randn(shape, dtype=torch.bfloat16, requires_grad=False).add(0.5).to('cuda')

# torch SDPA requires head_dim and head_dim_v to be the same
def _get_qkvo(bs, nh, lq, lkv, head_dim, head_dim_v=None):
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    q = generate_input(bs, nh, lq, head_dim)
    k = generate_input(bs, nh, lkv, head_dim_v)
    v = generate_input(bs, nh, lkv, head_dim_v)
    o = torch.empty((bs, nh, lq, head_dim_v), dtype=torch.bfloat16).to('cuda')
    return q, k, v, o

def _run_test_impl_torch(bs, nh, lq, lkv, head_dim, sdp_backend):
    t = 'SDP.cudnn' if sdp_backend == SDPBackend.CUDNN_ATTENTION else 'SDP.flash'
    tag = f'{t} bs{bs} nh{nh} lq{lq} lkv{lkv} head_dim{head_dim}'
    q, k, v, _ = _get_qkvo(bs, nh, lq, lkv, head_dim)

    with sdpa_kernel([sdp_backend]):
        time_ms = do_bench(lambda: F.scaled_dot_product_attention(q, k, v))
    tflops = get_tflops(bs, nh, lq, lkv, head_dim, head_dim, time_ms)
    print(f'\n[{tag}] t={time_ms}ms, TFLOPS={tflops}')
    assert lq == lkv, 'only one seqlen allowed'
    OUTPUT_FILE.write(f"{bs},{nh},{lq},{head_dim},{time_ms},{tflops},{t}\n")

def test_cudnn(seqlen=1024, dim=64):
    _run_test_impl_torch(4, 16, seqlen, seqlen, dim, SDPBackend.CUDNN_ATTENTION)

def test_flash(seqlen=1024, dim=64):
    _run_test_impl_torch(4, 16, seqlen, seqlen, dim, SDPBackend.FLASH_ATTENTION)


print('DIM=64 ############################')
test_cudnn(512)
test_cudnn(1024)
test_cudnn(2048)
test_cudnn(4096)
test_cudnn(8448)
test_cudnn(16384)
print('DIM=128 ##########################')
test_cudnn(512, 128)
test_cudnn(1024, 128)
test_cudnn(2048, 128)
test_cudnn(4096, 128)
test_cudnn(8448, 128)
test_cudnn(16384, 128)

OUTPUT_FILE.close()