## Intra-WG overlap
BEFORE

```python
n_block = n_block_max - 1
load_q()
load_k(n_block_max - 1)
load_v(n_block_max - 1)
for i in cutlass.range(n_block_max - 1):
    n_block = n_block_max - 2 - i # n_block_max-2, ..., 0
    load_k(n_block)
    load_v(n_block)
```

AFTER

```python
n_block = n_block_max - 1
load_q()
load_k(n_block_max - 1)
for i in cutlass.range(n_block_max - 1):
    n_block = n_block_max - 1 - i # n_block_max-1, ..., 1
    load_k(n_block - 1) # n_block_max-2, ..., 0
    load_v(n_block) # n_block_max-1, ..., 1

load_v(0)

```

HUGE BUG: If there's ever a concurrency issue, one reason might be forgetting to zero memory
- Not really a concurrency issue, but you get varying results everytime which SUGGESTS it's a concurrency issue.
- If the issue typically happens on later blocks, that's a clue
- For matmul, try testing multiplication of ones matrices, if you get only 2x, 4x etc. jumps that means it's accumulating on top of already accumulated memory
- Just don't forget this could be a problem, since concurrency typically leads me to think about barriers, and not this.

## Pingponging
No intra-wg-overlap
- before any MMA, warpgroup1 arrives at its own barrier
- all threads then barrier(so only warpgroup1 gets to proceed for now)
- after WG1 finishes QK, it arrives at WG2, kicking off it's MMA
- (let's say there's only 2 WGs) after WG1 finishes softmax it waits, WG2 arrives once it's done QK
- so after, WG1 will do PV and next QK, while WG2 does softmax, and then they swap

Yes intra-wg-overlap
- both warps do QK0 and softmax0
- first barrier is starting at QK1 in mainloop, so WG1 proceeds with QK1, PV0, and then arrives
- second warpgroup starts doing that, first warpgroup does softmax
- then, you swap roles