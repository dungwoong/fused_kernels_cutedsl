import argparse
from attn import FlashSM90, convert_from_dlpack
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import cuda.bindings.driver as cuda
import time
import cutlass
from cutlass import cute

# ncu --set full -o kernel -f python3 ncu.py
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ncu run ONLY")
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--lq", type=int, default=8192)
    parser.add_argument("--lk", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    q, k, v, o = _get_qkvo(args.bs, args.h, args.lq, args.lk, args.dim, args.dim)
    rsqrt_d = args.dim ** -0.5

    [q_cute, k_cute, v_cute, o_cute] = [convert_from_dlpack(x) for x in (q, k, v, o)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    fa = FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=True, pingpong=True)
    compiled_fa = cute.compile(fa, q_cute, k_cute, v_cute, o_cute, rsqrt_d, current_stream)
    compiled_fa(q_cute, k_cute, v_cute, o_cute, rsqrt_d, current_stream)

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        time.sleep(1)
        F.scaled_dot_product_attention(q, k, v)
    
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        time.sleep(1)
        F.scaled_dot_product_attention(q, k, v)
