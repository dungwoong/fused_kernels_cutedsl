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

# Modifying the epilogue
`accumulators : tensor<ptr<f32, rmem, align<32>> o ((2,2,32),1,1):((1,2,4),0,0)>`
- finished adding the elementwise rsqrt and sum to the sum, and then you just do a multiplication with each element in the accumulators

## Results
```
my_ms=0.19207999855279922, other_ms=0.2006400004029274
my_flops=715.5297506638651, other_flops=685.0027571570655
FLOPs numbers are incorrect since I only used the GEMMs flops but honestly RMS flops are nothing compared to GEMM

max_incorrect : 2.0
max_rel_incorrect : 26.5
```
- For some test randn matrices, we have ~40000 elements that were had an absolute error of >1. 1 is crazy work
- I should maybe double-check that all my stuff is correct, make sure sum is broadcasting properly. Otherwise, this is kinda what we would expect.
- I thought casting to fp32 would help precision though, but I'm guessing torch rmsnorm or whatever also casts to fp32

Next steps, figure out what to do about precision and then I can probably first test other kernels before working on optimization. RMSNorm seems like a bad candidate.

Also, anything I can do to make my work generalize better since I'm just manually hacking layouts at this point