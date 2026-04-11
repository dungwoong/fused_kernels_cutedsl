from gemm import *

if __name__ == "__main__":
    print('Starting...')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["debug", "speed", "ncu"])
    args = parser.parse_args()
    IS_NCU = args.mode == 'ncu'
    IS_DEBUG = args.mode == 'debug'
    IS_SPEED = args.mode == 'speed'

    m, n, k = 65536, 4096, 8192
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
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    # a_cute, b_cute, c_cute = [convert_from_dlpack(x) for x in (a, b, c)]
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
    compiled_gemm(a, b, c, current_stream)
    if not IS_NCU:
        print('All close:', torch.allclose(ref, c))
    if IS_DEBUG:
        print(ref)
        print(c)

    if IS_DEBUG:
        n_incorrect = c.numel() - ((c - ref).abs() < 0.001).sum()
        print('n_incorrect :', n_incorrect)
        print('n_nonzero :', (c != 0).sum())

    @torch.compile
    def torch_gemm():
        return a @ b.t()
    
    def cdsl_func(a, b):
        o = torch.empty(a.shape[0], b.shape[0], dtype=torch.bfloat16, device='cuda')
        compiled_gemm(a, b, o, current_stream)
        return o

    if IS_SPEED:
        my_ms = do_bench(lambda: cdsl_func(a, b))
        other_ms = do_bench(torch_gemm)
        print(f'{my_ms=}, {other_ms=}')
        my_flops, other_flops = get_tflops(my_ms), get_tflops(other_ms)
        print(f'{my_flops=}, {other_flops=}')