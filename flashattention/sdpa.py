import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import functools
import statistics
import time
import math
from triton import runtime

def profile_ms(op, repeats=30):

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
        start[i].record()
        op()
        end[i].record()

    torch.cuda.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in zip(start, end)])

def get_flops(bs, nh, lq, lkv, head_dim, latency_ms):
    return 4 * bs * nh * lq * lkv * head_dim / latency_ms / 1e9

# if output has a mean of 0, we get a large relative error
def generate_input(*shape):
    return torch.randn(shape).add(0.5).bfloat16().cuda()

# reimplementation of what we want to happen with attention
def attn_reimpl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    p = q @ k.transpose(2, 3)
    p = p * ((q.size(-1)**-0.5) * math.log2(math.e))
    p = torch.exp2(p - torch.max(p, dim=-1, keepdim=True).values)
    return (p @ v) / torch.sum(p, dim=-1, keepdim=True)

def run(bs, nh, lq, lkv, head_dim):
    Q = generate_input(bs, nh, lq, head_dim)
    K = generate_input(bs, nh, lkv, head_dim)
    V = generate_input(bs, nh, lkv, head_dim)

    output = F.scaled_dot_product_attention(Q, K, V)
    reimpl = attn_reimpl(Q, K, V)
    print(output.shape)
    
    print(output - reimpl)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        time.sleep(1)
        flash_ms = profile_ms(lambda: F.scaled_dot_product_attention(Q, K, V))
        flash_flops = get_flops(bs, nh, lq, lkv, head_dim, latency_ms=flash_ms)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        time.sleep(1)
        cudnn_ms = profile_ms(lambda: F.scaled_dot_product_attention(Q, K, V))
        cudnn_flops = get_flops(bs, nh, lq, lkv, head_dim, latency_ms=cudnn_ms)

    print(f'{flash_ms=}, {cudnn_ms=}')
    print(f'{flash_flops=}, {cudnn_flops=}')

run(16, 16, 1024, 1024, 64)


