- Cannot print from producer, need more registers. If you try to debug and it suddenly fails, maybe you don't have enough registers
- I need a separate buffer for epilogue, can't reuse sQ anymore unless if you want extra syncing lol
- I have to do a sync BETWEEN the producer and consumer when the consumer enters the epilogue to wait until we no longer need memory. You could also just do the final consumer release after this, but we can just opt for a barrier instead.

- I have no idea why that last barrier is required...

Profiling
- ok so persistent can get up to 95% for the one matrix size I tested
- one thing is that intra warpgroup overlap slows your kernel down, only becomes useful when you have 3 stages. I can look into that.
- I found (128, 256) doesn't actually speed things up, this kernel is more about pipelining since it has high CI already I would imagine

A bug
- it specifically has to do with persistence
- FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=True, pingpong=False)
- num stages must be >=3 and intra wg overlap must be True, also seqlen must be 2048 or less
- it happens when we run the kernel repeatedly, the kernel can run and is correct, but when we profile it crashes

Speeds
- 4, 16, 512, 64: FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=False, pingpong=True) gives you ~98%


## Inspecting one specific shape
```
Mine:  498.15294818301595 TFLOPS (0.5604159832000732)
Torch: 553.1263846538113 TFLOPS (0.5047180571845759)
0.90061324500873
```

- 2, 8, 8192, 64, I did: FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=True, pingpong=True)
- My memory and compute throughput is slightly lower than the reference.

Memory Throughput
- I have high **returns to sm**(memory tables, L1). I don't know what that means but that's been a persistent issue
- traffic from L2 cache to SMEM and vice versa using TMA is lower than ref. The reference kernel uses global loads and stores, and we use a lot more TMA instructions
- the reference has a lot of FADD and FMUL instructions, maybe from manual address calculation
- from instruction mix, it seems like I have a lot more movement between vector and uniform registers
- They use all the SMEM, so I'll do that too

- I have +95% branch instructions, might be difficult to avoid unless if
- I can look at source counters to try to fine-tune things