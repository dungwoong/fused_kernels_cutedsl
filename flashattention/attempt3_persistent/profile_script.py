from attn import FlashSM90, convert_from_dlpack
import torch
import torch.nn.functional as F
from cutlass import cute
import cuda.bindings.driver as cuda
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import do_bench
from functools import partial
import math
import multiprocessing as mp
import io
import sys
import traceback


def get_qkvo(bs, h, seqlen, dim):
    q = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    k = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    v = torch.randn((bs, h, seqlen, dim), dtype=torch.bfloat16).add(0.5).to('cuda')
    o = torch.ones((bs, h, seqlen, dim), dtype=torch.bfloat16).to('cuda')
    return q, k, v, o

def get_tflops(b, h, seqlen, dim, latency_ms):
    qk = b * h * (2 * seqlen * seqlen * dim)
    smx = b * h * (5 * seqlen * seqlen) # max + sub + exp + sum + div, but codebases(e.g. thunderkittens) use 4
    kv = b * h * (2 * seqlen * seqlen * dim)
    return (qk + smx + kv) / latency_ms / 1e9

def run_torch(q, k, v):
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        time_torch = do_bench(lambda: F.scaled_dot_product_attention(q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v)
    return ref, time_torch

CONFIGS_64 = (
    [256, 2, False, False],
    [256, 2, False, True],
    [128, 2, False, False],
    [128, 2, True, False],
    [128, 2, False, True],
    [128, 2, True, True],
    [128, 3, False, False],
    [128, 3, False, True],
    [128, 3, True, False],
    [128, 3, True, True],
)

CONFIGS_128 = (
    [128, 2, False, False],
    [128, 2, False, True],
    [128, 2, True, False],
    [128, 2, True, True], # Idk if you can do 3 here
)

def get_str(method, problem_dims, mma_qk_n=None, num_stages=None, iwo=None, pingpong=None, correct=None, ms=None, tflops=None, error=None, torch_time=None):
    problem_dims = ','.join(str(d) for d in problem_dims)
    return f"{method},{problem_dims},{mma_qk_n},{num_stages},{iwo},{pingpong},{correct},{ms},{tflops},{error},{torch_time}\n"

def run_cute(bs, h, seqlen, dim, mma_qk_n, num_stages, iwo, pingpong, torch_time):
    STREAM = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    tflops = partial(get_tflops, bs, h, seqlen, dim)
    rt = 1 / math.sqrt(dim)
    q, k, v, o = get_qkvo(bs, h, seqlen, dim)
    [q_cute, k_cute, v_cute, o_cute] = [convert_from_dlpack(x) for x in (q, k, v, o)]
    o_ref = F.scaled_dot_product_attention(q, k, v)
    fa = FlashSM90(qk_mn=(128, mma_qk_n), num_stages=num_stages, cluster_size_m=1, intra_wg_overlap=iwo, pingpong=pingpong)
    compiled_fa = cute.compile(fa, q_cute, k_cute, v_cute, o_cute, rt, STREAM)
    compiled_fa(q_cute, k_cute, v_cute, o_cute, rt, STREAM)
    correct = torch.allclose(o_ref, o, atol=1e-1, rtol=1e-1)
    t_cute = do_bench(lambda: compiled_fa(q_cute, k_cute, v_cute, o_cute, rt, STREAM))
    f_cute = tflops(t_cute)
    return get_str("cute", (bs, h, seqlen, dim), mma_qk_n, num_stages, iwo, pingpong, correct, t_cute, f_cute, torch_time=torch_time)

def async_wrapper(queue, bs, h, seqlen, dim, mma_qk_n, num_stages, iwo, pingpong, torch_time):
    buf = io.StringIO()
    old_out, old_error = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        torch.cuda.init()
        output = run_cute(bs, h, seqlen, dim, mma_qk_n, num_stages, iwo, pingpong, torch_time)
        queue.put(('ok', output, ""))
    except Exception as e:
        err_str = "\"" + str(e) + "\""
        queue.put(('error', get_str("cute", (bs, h, seqlen, dim), error=err_str), traceback.format_exc()))
    finally:
        sys.stdout, sys.stderr = old_out, old_error

def run_test(f, bs, h, seqlen, dim, mma_qk_n, num_stages, iwo, pingpong, torch_time, timeout=30):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=async_wrapper, args=(q, bs, h, seqlen, dim, mma_qk_n, num_stages, iwo, pingpong, torch_time))
    p.start()

    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        payload = get_str("cute", (bs, h, seqlen, dim), error="timeout")
    else:
        status, payload, logs = q.get(timeout=timeout)
    f.write(payload)
    f.flush()
    print(payload)

def run_trial(f, bs, h, seqlen, dim):
    """
    outputs will be method, bs, h, seqlen, dim, correct, mma_qk_n, num_stages, iwo, pingpong, ms, tflops
    mma_qk_m is always 128
    """
    rt = 1 / math.sqrt(dim)
    dims = (bs, h, seqlen, dim)
    q, k, v, o = get_qkvo(bs, h, seqlen, dim)
    tflops = partial(get_tflops, bs, h, seqlen, dim)

    # Run torch
    o_ref, t_torch = run_torch(q, k, v)
    f_torch = tflops(t_torch)
    f.write(get_str("torch", dims, correct=True, tflops=f_torch, ms=t_torch))
    print(f'torch time {t_torch}')

    # [q_cute, k_cute, v_cute, o_cute] = [convert_from_dlpack(x) for x in (q, k, v, o)]
    configs = CONFIGS_64 if dim == 64 else CONFIGS_128
    for i, (mma_qk_n, num_stages, iwo, pingpong) in enumerate(configs):
        print(f'running {mma_qk_n}, {num_stages}, {iwo}, {pingpong}')
        run_test(f, bs, h, seqlen, dim, mma_qk_n, num_stages, iwo, pingpong, t_torch)
        # o.fill_(1) # so we can re-verify
        # fa = FlashSM90(qk_mn=(128, mma_qk_n), num_stages=num_stages, cluster_size_m=1, intra_wg_overlap=iwo, pingpong=pingpong)
        # compiled_fa = cute.compile(fa, q_cute, k_cute, v_cute, o_cute, rt, STREAM)
        # compiled_fa(q_cute, k_cute, v_cute, o_cute, rt, STREAM)
        # correct = torch.allclose(o_ref, o, atol=1e-1, rtol=1e-1)
        # if not correct:
        #     print(f"WARNING: Incorrect")
        # t_cute = do_bench(lambda: compiled_fa(q_cute, k_cute, v_cute, o_cute, rt, STREAM))
        # f_cute = tflops(t_cute)
        # f.write(f"cute,{dims_str},{mma_qk_n},{num_stages},{iwo},{pingpong},{correct},{t_cute},{f_cute}\n")


if __name__ == "__main__":
    """
    'hidden dimension' is dim * nheads
    'head dimension' is just dim
    """
    print("starting...")
    total_tokens = 16384
    with open("data.csv", "w") as f:
        for dim in (64, 128):
            for h in (8, 16):
                for seqlen in (512, 1024, 2048, 4096, 8192, 16384):
                    b = total_tokens // seqlen
                    print(f'Starting {b}, {h}, {seqlen}, {dim}')
                    run_trial(f, b, h, seqlen, dim)
    print('Done')
