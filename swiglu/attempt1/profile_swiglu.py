import torch
from gemm import GemmSM90
import cuda.bindings.driver as cuda
from cdsl_fn_utils import jit_cache, STREAM, convert_from_dlpack, make_fake_tensor
from profile_run import run_experiment
import cutlass
from cutlass import cute
import math
from functools import partial


@jit_cache
def _compile_swiglu(m, n, k):
    gemm = GemmSM90(tile_shape_mn=(128, 128), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=False,
                    is_persistent=True,
                    gemm_n_prologue=0)
    dtype = cutlass.BFloat16
    # batch_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, k)
    divn = math.gcd(128 // dtype.width, n)
    x = make_fake_tensor(dtype, (m, k), div)
    w = make_fake_tensor(dtype, (n, k), div)
    v = make_fake_tensor(dtype, (n, k), div)
    o = make_fake_tensor(dtype, (m, n), div)
    return cute.compile(gemm, x, w, v, o, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")

def _swiglu_fwd(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor, o: torch.Tensor):
    mnk = (o.shape[0], o.shape[1], x.shape[1]) # mnk
    # x_cute, w_cute, v_cute, o_cute = [convert_from_dlpack(t) for t in (x, w, v, o)]
    _compile_swiglu(*mnk)# (x, w, v, o)

# input tensors are (m, k) and (n, k)
def swiglu(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device='cuda')
    _swiglu_fwd(x, w, v, out)
    return out

def generate_tensors(m, n, k):
    mul_factor = 1 / k**0.5
    x = torch.randn((m, k), dtype=torch.bfloat16).mul(mul_factor).to('cuda').detach() # 1/sqrt(d) scaling
    w = torch.randn((n, k), dtype=torch.bfloat16).mul(mul_factor).to('cuda').detach()
    v = torch.randn((n, k), dtype=torch.bfloat16).mul(mul_factor).to('cuda').detach()
    return (x, w, v)

@torch.compile
def torch_swiglu_simple(x, w, v):
    in1 = x @ w.t()
    in2 = x @ v.t()
    return torch.nn.functional.silu(in1) * in2

def cdsl(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor, compiled) -> torch.Tensor:
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device='cuda')
    compiled(x, w, v, out)
    return out

if __name__ == '__main__':
    compiled = _compile_swiglu(4096, 4096, 4096)
    run_experiment(generate_tensors, torch_swiglu_simple, swiglu)