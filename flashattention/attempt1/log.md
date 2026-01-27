## Initial FA
- QK is bundled for the first iter, so you have extra tma bytes. You could just as easily have a new barrier for Q, but it doesn't matter.
- They created a new PipelineTmaAsync class that allows extra_tx_count
    - They set the `__class__` to `PipelineTmaAsync` so for future comparisons everything will work as if it's a tma async
    - they just have an extra available argument for `producer_acquire` so when you load q you can augment the tx count
- Ran into some slicing/layout issues with loading Q(since it's single stage)

## Inter-wg stuff
- Sometimes people don't even explicitly add this(e.g. cutedsl version)
- in the mainloop c code, look for warp_scheduler_barrier_sync and arrive. Look for `FwdNamedBarriers`

## Accumulation
- O_should_accumulate = False and then set to true right after the first accumulation
- they have a `warp_scheduler_barrier_arrive` fn that does the ping-ponging.

## Pingponging
- they allow ping-ponging with up to 3 warpgroups somehow
- right after dispatching qkgemm, they arrive at barrier. Before doing PV, they sync at barrier
- You arrive at the barrier of the next WG e.g. WG0 --> 1 --> 2 --> 0
- You then sync at your own barrier
- So overall, once you do gemm0, the next warp can do gemm0
- In `mma_barrier_init`, they arrive at WG1's barrier so when they call sync later, WG1 immediately is good to go

## Online softmax
- IMPORTANT: if we had two atoms side by side, then the online softmax would fail, since it assumes warps hold the entire row
- We should just think of some way to annotate things so that this doesn't happen
- so basically you run online_softmax to get new row scale and row max, then rescale_o
- at the end of your accumulation, you run finalize() to get the reciprocal and then rescale o again
- they have a compiletime is_first for softmax to do stuff so you HAVE to put the first iter outside the loop, and then do next iters later. IMPORTANT: need to think how this factors into our compiler but yeah.

## Adding the epilogue
- Reuse sQ data iterator, sO might be larger but we have multiple stages.

## Locations
- For loop over `n_tile`: 2132 and surrounding
- One loop helper fn: 2302 `mma_one_n_block`

Softmax can run, need to see if it's correct