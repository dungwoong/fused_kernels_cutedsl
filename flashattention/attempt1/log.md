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

## Adding the epilogue
- Reuse sQ data iterator, sO might be larger but we have multiple stages.

TODO make sure loaded tile is correct(we can probably check this later, it should be fine though)