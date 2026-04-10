import torch
import argparse
from gemm import GemmSM90
import cuda.bindings.driver as cuda
from triton.testing import do_bench
import time

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # good idea?


def validate(expected, out):
    expected = expected.float()
    out = out.float()
    diff = (out - expected).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (expected.abs().clamp(min=1.0))).max().item()
    return max_abs, max_rel


# tensors[-1] must be the output tensor
def run_experiment(generate_tensors, torch_ref, cutedsl_kernel, generate_torch_tensors=None):
    parser = argparse.ArgumentParser(description="Profiling Program For Gemm-Based Kernel")
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--workload", type=str, default="unknown")
    parser.add_argument("--output_keys_only", type=bool, default=False)
    args = parser.parse_args()
    output = {
        "workload": args.workload,
        "atol": args.atol,
        "rtol": args.rtol,
        "time_ms": None, 
        "allclose": None,
        "max_abs": None,
        "max_rel": None,
    }

    def print_keys():
        output_str = ','.join(str(k) for k in output.keys())
        print(output_str)
    def print_output():
        output_str = ','.join(str(v) for v in output.values())
        print(output_str)
    
    if args.output_keys_only:
        print_keys()
        return
    
    # try:
    tensors = generate_tensors(args.m, args.n, args.k)
    torch_tensors = tensors
    if generate_torch_tensors is not None:
        torch_tensors = generate_torch_tensors(*tensors)
    # cdsl_tensors = [convert_from_dlpack(t) for t in tensors]
    # torch_tensors = tensors[:-1]

    # compiled_cdsl = cute.compile(*cdsl_tensors, **cdsl_kwargs)
    # compiled_cdsl(*cdsl_tensors, **cdsl_kwargs)

    ref = torch_ref(*torch_tensors)
    o = cutedsl_kernel(*tensors)

    output['allclose'] = torch.allclose(ref, o, atol=args.atol, rtol = args.rtol)
    output['max_abs'], output['max_rel'] = validate(ref, o)

    # timing
    time.sleep(1) # ??
    output['time_ms'] = do_bench(lambda: cutedsl_kernel(*tensors))
    # except Exception as e:
    #     print(e)
    # finally:
    print_output()

