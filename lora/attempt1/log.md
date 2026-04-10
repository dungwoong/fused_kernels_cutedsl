# Adding LoRA
- if the column is 0, then we must calculate the A output tile
- Could be a problem if A is too big, cuz it will take up so much resources in the first wave and then you can't use that SMEM for anything, unless if we manage to repurpose the SMEM


As a gemm epilogue, we're going to load in a tile of xA, a tile of B and multiply them into our thing.
e.g. xA size is 128x16, xB size is 256x16
We can try varying the sizes from 4 to 64. Let's start with multiples of 16 to replicate the performance.