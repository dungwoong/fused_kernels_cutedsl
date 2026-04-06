[(Wx) x sigmoid(Wx)] x (Vx) or alternatively silu(Wx) x Vx

# Some feedback for the auto-generation
- Slicing tma tensors is going into the loop, and it's not being used by the load for some reason.
- For the producer, you can switch the `if warp_idx == 0` and the `for` loop

# Modifying GEMM.py
- before, supports ab. Now, we want xW, xV so we need A = x and then we need to support W and V so there's just 2 B matrices
- gonna add `b1=v` and assign `b=w`