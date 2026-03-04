- trying more warpgroups isn't working since we get stuck on warpgroup reg alloc
- I can just try adding multicasting probably
- we can also do the precomputation and then store back to SMEM maybe?

`FlashSM90(qk_mn=(128, 128), num_stages=2, cluster_size_m=1, intra_wg_overlap=False, pingpong=False, mma_m_size=64)` for `2, 8, 512, 64` beats cudnn when you set registers to (24, 160) instead of (24, 240)

`FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=True, pingpong=True, mma_m_size=64)` for (2, 8, 8192, 64) is 86% cudnn

- "if a setmaxnreg instruction is not executed by all warps in the warpgroup, then behavior is undefined" [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)
- we have branches for each load instruction, instead of just like 1 big branch. I wonder if that's not ideal...
- it elects a leader so many times, it's not good but ok
- I have mul and sub at like 620-648, I wonder if I could use FMA


- https://github.com/NVIDIA/cutlass/issues/2981


Barrier update
- the epilogue initial barrier is not required. stmatrix is synced within the warp anyways. That second barrier makes sure all warps have finished.
- the first barrier is hacked in because we lumped q and k together. Think about it, the producer will drop off k[n] early and then check the next index that previously corresponded to k[n-stages]. If it's good, then it's going to load in the next q. So there's no ordering that q must be loaded in AFTER k[n], it actually corresponds to k[n-stages] and that's why we get a race condition. So next time just put in a barrier lol