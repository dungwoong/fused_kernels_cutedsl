[(Wx) x sigmoid(Wx)] x (Vx) or alternatively silu(Wx) x Vx

# Some feedback for the auto-generation
- Slicing tma tensors is going into the loop, and it's not being used by the load for some reason.
- For the producer, you can switch the `if warp_idx == 0` and the `for` loop

# Modifying GEMM.py
- before, supports ab. Now, we want xW, xV so we need A = x and then we need to support W and V so there's just 2 B matrices
- gonna add `b1=v` and assign `b=w`
- We need to actually consider kaiming initialization or something or else when you multiply the results of two 4096 gemms...yeah the outputs are like 16k, errors are like 128. It's bad.

# Things to modify
- Add a separate pipeline for V because we don't need to wait for it with W
- We could try something like holding X in registers so we don't need to re-read it from SMEM. Should consult Nsight compute for this though, it might already be doing it(?)

- Separate pipeline for V didn't really work. Tensor cores are already full so separate pipeline doesn't do much, it can actually slow things down since more barrier synchronization.
- If we have high arithmetic intensity, we could go with a ping-pong strategy to sacrifice some arithmetic intensity for epi/mainloop overlap