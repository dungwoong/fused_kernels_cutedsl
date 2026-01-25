## Initial FA
- QK is bundled for the first iter, so you have extra tma bytes. You could just as easily have a new barrier for Q, but it doesn't matter.
- They created a new PipelineTmaAsync class that allows extra_tx_count
    - They set the `__class__` to `PipelineTmaAsync` so for future comparisons everything will work as if it's a tma async
    - they just have an extra available argument for `producer_acquire` so when you load q you can augment the tx count
- Ran into some slicing/layout issues with loading Q(since it's single stage)

TODO make sure loaded tile is correct(we can probably check this later, it should be fine though)