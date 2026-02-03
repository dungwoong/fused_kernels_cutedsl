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
