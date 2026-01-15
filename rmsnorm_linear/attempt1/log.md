## first step is changing it from smem to register MMA
- Do you need a proxy fence after? [I don't think so](https://forums.developer.nvidia.com/t/why-arent-there-explicit-async-proxy-generic-proxy-fences-in-the-cuda-guide-tma-prefetching-example/357574)
- We can look at sm120 stuff too, since they have to use ldmatrix I think
- Doing a bunch of stuff with retiling etc. was good, since we're just addressing the A matrix here. In the future we might have to copy quack
- I took out async MMA but we'll see what we can pipeline later

## Adding warp reduction
```python
a_regs : tensor<ptr<bf16, rmem, align<32>> o ((8,1),1,4):((1,0),0,8)> # goes into the copy
a_regs0 : tensor<ptr<bf16, rmem, align<32>> o ((2,2,2),1,4):((1,2,4),0,8)> # is used for the WGMMA, 4 stages
```
- I manually made row reduce layout
- The row reduce algorithm is also quite manual too, it's just a 3-nested loop

## TODOs
- Warp reduce(you just need to get 4 threads per row to get the items)
- When writing epilogue, you have to do divisions