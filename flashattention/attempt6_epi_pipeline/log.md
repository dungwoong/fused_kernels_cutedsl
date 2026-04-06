# How does the epilogue work?
- You arrive at a barrier
- Make a copy atom stmatrix x4 so you can store 16x16 tiles
- make tiled copy C atom, that should work
- Get your output tile
- partition it up with a thr copy. your epi smem layout should have stages to begin with btw so the partition should just give you stages
- ok then you gotta somehow partition up your SMEM and GMEM
- then you do the copy and yeah ok

- multicasting actually pushes up performance now, we can try that more often
- we can try staging q now to get more performance increase