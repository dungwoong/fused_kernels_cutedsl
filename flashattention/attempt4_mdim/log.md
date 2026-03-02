- trying more warpgroups isn't working since we get stuck on warpgroup reg alloc
- I can just try adding multicasting probably
- we can also do the precomputation and then store back to SMEM maybe?

`FlashSM90(qk_mn=(128, 128), num_stages=2, cluster_size_m=1, intra_wg_overlap=False, pingpong=False, mma_m_size=64)` for `2, 8, 512, 64` beats cudnn when you set registers to (24, 160) instead of (24, 240)

`FlashSM90(qk_mn=(128, 128), num_stages=3, cluster_size_m=1, intra_wg_overlap=True, pingpong=True, mma_m_size=64)` for (2, 8, 8192, 64) is 86% cudnn