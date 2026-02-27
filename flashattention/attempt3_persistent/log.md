- Cannot print from producer, need more registers. If you try to debug and it suddenly fails, maybe you don't have enough registers
- I need a separate buffer for epilogue, can't reuse sQ anymore unless if you want extra syncing lol
- I have to do a sync BETWEEN the producer and consumer when the consumer enters the epilogue to wait until we no longer need memory. You could also just do the final consumer release after this, but we can just opt for a barrier instead.

- I have no idea why that last barrier is required...

Profiling
- ok so persistent can get up to 95% for the one matrix size I tested
- one thing is that intra warpgroup overlap slows your kernel down, only becomes useful when you have 3 stages. I can look into that.
- I found (128, 256) doesn't actually speed things up, this kernel is more about pipelining since it has high CI already I would imagine