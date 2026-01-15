## first step is changing it from smem to register MMA
- Do you need a proxy fence after? [I don't think so](https://forums.developer.nvidia.com/t/why-arent-there-explicit-async-proxy-generic-proxy-fences-in-the-cuda-guide-tma-prefetching-example/357574)
- We can look at sm120 stuff too, since they have to use ldmatrix I think
- Doing a bunch of stuff with retiling etc. was good, since we're just addressing the A matrix here. In the future we might have to copy quack
- I took out async MMA but we'll see what we can pipeline later