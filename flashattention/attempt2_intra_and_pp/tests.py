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

# I don't have pytest on my sif so I'll try something manual

def get_tflops(bs, nh, lq, lkv, head_dim, head_dim_v, latency_ms):
    qk = bs * nh * (2 * lq * lkv * head_dim)
    smx = bs * nh * (4 * lq * lkv) # max + sub + exp + sum + div, but codebases(e.g. thunderkittens) use 4
    kv = bs * nh * (2 * lq * lkv * head_dim_v)
    return (qk + smx + kv) / latency_ms / 1e9

# if output has a mean of 0, we get a large relative error
def generate_input(*shape):
    return torch.randn(shape, dtype=torch.bfloat16).add(0.5).to('cuda')

# torch SDPA requires head_dim and head_dim_v to be the same
def _get_qkvo(bs, nh, lq, lkv, head_dim, head_dim_v=None):
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    q = generate_input(bs, nh, lq, head_dim)
    k = generate_input(bs, nh, lkv, head_dim_v)
    v = generate_input(bs, nh, lkv, head_dim_v)
    o = torch.empty((bs, nh, lq, head_dim_v), dtype=torch.bfloat16).to('cuda')
    return q, k, v, o

def profile_ms(op, repeats=30):
    stream = torch.cuda.current_stream()

    clear_cache = functools.partial(
        runtime.driver.active.clear_cache,  # type: ignore[attr-defined]
        runtime.driver.active.get_empty_cache_for_benchmark(),  # type: ignore[attr-defined]
    )
    clear_cache()

    # warmup
    op()
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        clear_cache()
        start[i].record(stream)
        op()
        end[i].record(stream)

    torch.cuda.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in zip(start, end)])

convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
    )

# just run pytest -s to collect outputs
# TODO this does not work for hung outputs on the GPU, so beware...
def _run_test_impl(fa, bs, nh, lq, lkv, head_dim, head_dim_v=None, tag="attn"):
    q, k, v, o= _get_qkvo(bs, nh, lq, lkv, head_dim, head_dim_v)
    q_cute, k_cute, v_cute, o_cute = [convert_from_dlpack(x) for x in (q, k, v, o)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    inv_sqrt_d = q.shape[-1]**-0.5
    compiled_fa = cute.compile(fa, q_cute, k_cute, v_cute, o_cute, inv_sqrt_d, current_stream)
    compiled_fa(q_cute, k_cute, v_cute, o_cute, inv_sqrt_d, current_stream)
    torch.cuda.synchronize()

    ref = F.scaled_dot_product_attention(q, k, v)
    
    # TODO consider choosing a different tolerance
    assert torch.allclose(ref, o, atol=1e-1, rtol=1e-2), f'Incorrect. Max abs diff: {torch.max((o - ref).abs()).item()}'
    
    time_ms = do_bench(lambda: compiled_fa(q_cute, k_cute, v_cute, o_cute, inv_sqrt_d, current_stream))
    print(f'\n[{tag}] t={time_ms}ms, TFLOPS={get_tflops(bs, nh, lq, lkv, head_dim, head_dim_v, time_ms)}')

def _run_test_impl_torch(bs, nh, lq, lkv, head_dim, sdp_backend):
    t = 'SDP.cudnn' if sdp_backend == SDPBackend.CUDNN_ATTENTION else 'SDP.flash'
    tag = f'{t} bs{bs} nh{nh} lq{lq} lkv{lkv} head_dim{head_dim}'
    q, k, v, _ = _get_qkvo(bs, nh, lq, lkv, head_dim)

    with sdpa_kernel([sdp_backend]):
        time_ms = do_bench(lambda: F.scaled_dot_product_attention(q, k, v))
    print(f'\n[{tag}] t={time_ms}ms, TFLOPS={get_tflops(bs, nh, lq, lkv, head_dim, head_dim, time_ms)}')


def async_wrapper(queue, fa, bs, nh, lq, lkv, head_dim, head_dim_v, tag):
    buf = io.StringIO()
    old_out, old_error = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        torch.cuda.init()
        _run_test_impl(fa, bs, nh, lq, lkv, head_dim, head_dim_v, tag)
        queue.put(('ok', buf.getvalue(), ""))
    except Exception as e:
        queue.put(('error', str(e), traceback.format_exc()))
    finally:
        sys.stdout, sys.stderr = old_out, old_error
    

def run_test(fa, bs, nh, lq, lkv, head_dim, head_dim_v, tag, timeout=30):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=async_wrapper, args=(q, fa, bs, nh, lq, lkv, head_dim, head_dim_v, tag))
    p.start()

    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        pytest.fail("Kernel hung") # TODO haven't tested this with actual hung gpu kernels, just time.sleep
    
    status, payload, logs = q.get(timeout=30)

    if status == 'error':
        pytest.fail(f"{payload}")
    print(f'{payload}, {logs}')

def test_cudnn_basic():
    _run_test_impl_torch(16, 16, 4096, 4096, 64, SDPBackend.CUDNN_ATTENTION)

def test_flash_basic():
    _run_test_impl_torch(16, 16, 4096, 4096, 64, SDPBackend.FLASH_ATTENTION)

def test_basic_cute():
    fa = FlashSM90(qk_mn=(128, 256), cluster_size_m=1)
    run_test(fa, 16, 16, 4096, 4096, 64, 64, 'basic_cute')

def test_basic_cute_128():
    fa = FlashSM90(qk_mn=(128, 256), cluster_size_m=1)
    run_test(fa, 16, 16, 4096, 4096, 128, 128, 'basic_cute128')

def test_basic_cluster():
    fa = FlashSM90(qk_mn=(128, 256), cluster_size_m=2)
    run_test(fa, 16, 16, 4096, 4096, 64, 64, 'basic_cluster')



# run pytest -s tests.py to collect print outputs