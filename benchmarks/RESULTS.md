# gradbridge benchmark

NVIDIA GeForce RTX 5090, torch 2.11.0+cu128, jax 0.12.0.1, tf 2.21.0, warp 1.19.0, Python 3.12.12; float32; median ms per forward+backward


## n = 1,000

| consumer | native | from numpy | from torch | from jax | from tf | from warp | hand-written |
|---|---|---|---|---|---|---|---|
| **torch** | 0.053 | 0.173 | 0.060 | 0.522 | 0.443 | 0.329 | warp: 0.329, jax: 1.235 |
| **jax** | 0.044 | 0.232 | 0.352 | 0.049 | 0.533 | 0.318 | — |
| **tf** | 0.260 | 0.815 | 0.938 | 1.248 | 0.268 | 0.860 | — |
| **warp** | 0.056 | 0.242 | 0.319 | 0.635 | 0.494 | 0.057 | — |

## n = 100,000

| consumer | native | from numpy | from torch | from jax | from tf | from warp | hand-written |
|---|---|---|---|---|---|---|---|
| **torch** | 0.057 | 0.683 | 0.063 | 0.603 | 0.483 | 0.378 | warp: 0.359, jax: 1.256 |
| **jax** | 0.050 | 0.509 | 0.328 | 0.054 | 0.499 | 0.318 | — |
| **tf** | 0.334 | 1.391 | 0.964 | 1.274 | 0.305 | 0.896 | — |
| **warp** | 0.058 | 0.648 | 0.312 | 0.640 | 0.512 | 0.059 | — |

## n = 10,000,000

| consumer | native | from numpy | from torch | from jax | from tf | from warp | hand-written |
|---|---|---|---|---|---|---|---|
| **torch** | 0.344 | 125.926 | 0.345 | 0.609 | 0.683 | 1.266 | warp: 1.269, jax: 1.290 |
| **jax** | 0.193 | 80.667 | 0.669 | 0.197 | 0.709 | 0.846 | — |
| **tf** | 6.580 | 133.155 | 7.179 | 7.113 | 6.361 | 7.334 | — |
| **warp** | 0.395 | 126.151 | 0.658 | 0.852 | 1.115 | 0.402 | — |
