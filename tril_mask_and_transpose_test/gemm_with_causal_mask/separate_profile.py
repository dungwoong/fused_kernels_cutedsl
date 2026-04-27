from gemm import *
from profile_utils import run_experiment, get_args

if __name__ == "__main__":
    print('Starting...')

    args = get_args()
    m, n, k = args.m, args.n, args.k
    flops = 2 * m * n * k

    def get_tflops(time_ms):
        return (flops / (time_ms / 1e3)) / 1e12

    dtype = cutlass.BFloat16
    div = math.gcd(128 // dtype.width, k)
    divn = math.gcd(128 // dtype.width, n)
    a = torch.randn((m, k), dtype=torch.bfloat16).to('cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16).to('cuda')
    c = torch.empty((m, n), dtype=torch.bfloat16).to('cuda')
    ref = a @ b.t()
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    gemm = GemmSM90(tile_shape_mn=(128, 256), 
                    epi_tile_mn=(128, 32),
                    cluster_shape_mnk=(2, 1, 1), 
                    atom_layout_mn=(2, 1),
                    ab_stage=3,
                    reuse_ab=False,
                    is_persistent=True)
    a_c = make_fake_tensor(dtype, (m, k), div)
    b_c = make_fake_tensor(dtype, (n, k), div)
    c_c = make_fake_tensor(dtype, (m, n), divn)
    compiled_gemm = cute.compile(gemm, a_c, b_c, c_c, current_stream, options='--enable-tvm-ffi')
    tensors = (a, b)
    @torch.compile
    def torch_gemm(a_, b_):
        return a_ @ b_.t()

    def cdsl_func(a_, b_):
        o = torch.empty(a_.shape[0], b_.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a_, b_, o, current_stream)
        return o
    
    run_experiment(args, tensors, torch_gemm, tensors, cdsl_func)
    # compiled_gemm(a, b, c, current_stream)
    # if not IS_NCU:
    #     print('All close:', torch.allclose(ref, c))
    # if IS_DEBUG:
    #     print(ref)
    #     print(c)

    # if IS_DEBUG:
    #     n_incorrect = c.numel() - ((c - ref).abs() < 0.001).sum()
    #     print('n_incorrect :', n_incorrect)
    #     print('n_nonzero :', (c != 0).sum())

    # if IS_SPEED:
    #     my_ms = do_bench(lambda: cdsl_func(a, b))
    #     other_ms = do_bench(torch_gemm)
    #     print(f'{my_ms=}, {other_ms=}')
    #     my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
    #     print(f'{my_flops=}, {other_flops=}')