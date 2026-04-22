- take flash attention mask, only consider `mask_causal`
- cute `make_identity_tensor` is helpful e.g. `tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]`
- so the idea is make identity tensor, use the tiled mma to make it into the right shape, and then mask using it


Notes
- you have to make sure the acc is the right layout
- you have to do `cutlass.Float32(...)` to fill with the correct datatype

Here is the debug info:

```
tile m n = 128 256 for 2 WGs
gemm_acc: tensor<ptr<f32, rmem, align<32>> o ((2,2,32),1,1):((1,2,4),0,0)>
gemm_acc_mn: tensor<ptr<f32, rmem, align<32>> o ((2,1),(2,32,1)):((2,0),(1,4,0))>
```

so in the acc, the first dim is actually the row. The second dim is the col.

# Looking at registers from ldmatrix

- for ldmatrix, it's different so we're doing m128n256k64 so it's loading in 64x64 btw

```
accumulators : tensor<ptr<f32, rmem, align<32>> o ((2,2,32),1,1):((1,2,4),0,0)>
a_regs: tensor<ptr<bf16, rmem, align<32>> o ((8,1),1,4):((1,0),0,8)>
a_regs_mma: tensor<ptr<bf16, rmem, align<32>> o ((2,2,2),1,4):((1,2,4),0,8)>
```

- `a_regs` is from the ldmatrix so the 8 is for the entire 16x16 and the 4 makes it `16 x 64`
- in `a_regs_mma` they change it to (2, 2, 2)
- ok well now we know what to target, I feel like we should probably target a_regs since what if the user wants to store back to smem or something?

# Debugging
- if you get an unspecified launch failure: one thing you can look at is whether you're advancing your states...