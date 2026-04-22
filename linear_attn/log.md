Not finished

# Test workloads
- In the GLA paper they go up to 2^15 which is ~30k tokens
- Wait, if you have evenly spaced chunks, then the actual causal mask is no problem. Workloads will be distributed evenly and it's a simple gemm operation
- So the problem is in the first kernel

# Kernel Design

Second kernel
- so you'll have the chunk size so you can grab the last chunk e.g. `chunk_size=64`, you know you can grab `0, 64, 128, etc.`
- so you just start at that index, go until the causal mask hits and then perform epilogue and end
- this should be pretty simple

First kernel
- params: chunk size, output tile shape is set at d, d, k split
- you have `b * h * num_chunks` outputs to save to, and your output should be `b, h, num_chunks, dim_m, dim_n`