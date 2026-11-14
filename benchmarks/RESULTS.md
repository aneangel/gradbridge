# gradbridge benchmarks

## Current optimization measurements

The latest recorded optimization audit used an RTX 5090, PyTorch 2.10.0+cu128,
JAX 0.12.0.1, TensorFlow 2.21.0 and Warp 1.16.0. The
[environment and CPU/GPU validation](../docs/downstream.md#tested-environment) record the
full configuration and 505 passing tests with zero skips.

| Optimization | Baseline and scope | Recorded speedup |
|---|---|---:|
| Eager PyTorch inference without source backward-state recording | Same checked downstream inference workload, automatic evaluation vs forced residual recording | CPU 1.09–1.28×; GPU 1.07–1.20× |
| Native PyTorch callbacks from JAX | Same current derivative rules, native GPU buffers vs forced host fallback, 1,048,576 float32 elements, value + gradient | Scalar 5.69×; batched 19.61×; five-dimensional 40.48× |
| Native JAX→TensorFlow graph + XLA | Default device callback in `tf.function` vs export and whole-function XLA, 1,048,576 float32 elements, value + gradient | GPU 5.68× |

These comparisons measure different optimizations. They exclude warmup and check outputs
before timing; value-and-gradient benchmarks also check full gradients. Results are medians
of alternating rounds, with each GPU call synchronized. They do not establish a universal
speedup for all framework combinations or training workloads.

See [downstream inference tables and methodology](../docs/downstream.md#inference-performance)
and the [native-callback and graph audit](../docs/limitations-audit.md#measured-effect-of-removing-host-fallbacks)
for raw latencies, comparison contracts and optimized-program evidence of GPU fusion.
Reproduce from a source checkout with the downstream dependencies and CUDA-enabled
frameworks installed:

```bash
python benchmarks/bench_inference.py --markdown inference-cpu.md
python benchmarks/bench_inference.py --device cuda --markdown inference-gpu.md
python benchmarks/bench_jax_native.py --markdown native-callbacks.md
python benchmarks/bench_tf_native.py --markdown native-tf.md --hlo optimized.hlo
```

For multiple frameworks sharing a GPU, use `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`TF_FORCE_GPU_ALLOW_GROWTH=true`; the newer benchmark scripts set these defaults.

## Framework callback matrix

Measured on 2026-11-13 with the current checkout: NVIDIA GeForce RTX 5090,
PyTorch 2.10.0+cu128, JAX 0.12.0.1, TensorFlow 2.21.0, Warp 1.16.0,
Python 3.12.13; float32; median milliseconds per forward + backward.

The function is `F(x, y) = sin(x) * y`, with supplied output cotangents. The JAX consumer
is jitted; the TensorFlow consumer in this matrix uses eager `GradientTape` and materializes
the input gradient with `.numpy()` to synchronize. That transfer is included in its timings.
This matrix uses default callback paths, not optional native JAX→TensorFlow graph lowering.
The script warms up and synchronizes framework work, then reports median per-call latency.
Reproduce with CUDA-enabled frameworks:

```bash
python benchmarks/bench.py --markdown callback-results.md
```

### n = 1,000

| consumer | native | from numpy | from torch | from jax | from tf | from warp | hand-written |
|---|---|---|---|---|---|---|---|
| **torch** | 0.055 | 0.183 | 0.061 | 0.573 | 0.454 | 0.308 | warp: 0.320, jax: 1.232 |
| **jax** | 0.049 | 0.230 | 0.322 | 0.044 | 0.474 | 0.308 | — |
| **tf** | 0.256 | 0.767 | 0.750 | 1.234 | 0.264 | 0.836 | — |
| **warp** | 0.055 | 0.248 | 0.257 | 0.599 | 0.503 | 0.056 | — |

### n = 100,000

| consumer | native | from numpy | from torch | from jax | from tf | from warp | hand-written |
|---|---|---|---|---|---|---|---|
| **torch** | 0.060 | 0.654 | 0.059 | 0.578 | 0.471 | 0.316 | warp: 0.414, jax: 1.231 |
| **jax** | 0.050 | 0.491 | 0.333 | 0.044 | 0.482 | 0.314 | — |
| **tf** | 0.318 | 1.252 | 0.792 | 1.237 | 0.334 | 0.926 | — |
| **warp** | 0.057 | 0.668 | 0.221 | 0.609 | 0.516 | 0.058 | — |

### n = 10,000,000

| consumer | native | from numpy | from torch | from jax | from tf | from warp | hand-written |
|---|---|---|---|---|---|---|---|
| **torch** | 0.346 | 130.896 | 0.348 | 0.650 | 0.691 | 1.192 | warp: 1.212, jax: 1.292 |
| **jax** | 0.191 | 79.721 | 0.634 | 0.189 | 0.701 | 0.810 | — |
| **tf** | 5.823 | 136.410 | 7.059 | 7.541 | 6.592 | 7.459 | — |
| **warp** | 0.397 | 127.748 | 0.566 | 0.872 | 1.104 | 0.398 | — |

## Warp simulation and policy workloads

Measured on 2026-11-13 with the current code and the environment above, including
Newton 1.6.0.dev0. Each cell is median milliseconds per iteration, without / with CUDA graphs:

| Workload | Native Warp | Hand-written bridge | gradbridge |
|---|---:|---:|---:|
| Newton contact rollout, 288 substeps, forward + backward | 67.800 / 4.040 | 70.057 / 4.110 | 69.226 / 4.180 |
| PyTorch MLP through 200-step Warp projectile, batch 4096, including Adam update | — | 15.47 / 1.90 | 15.18 / 1.79 |

Newton uses `cost_per_iteration()` from
[newton_ball_torch.py](../examples/newton_ball_torch.py): three warmup iterations, then
30 timed iterations per approach, synchronized on CUDA. All six approaches produced
bit-identical input gradients. `capture=True` reduced the measured gradbridge iteration
by about 17×; the native and hand-written graph implementations have similar costs.

The projectile measurement runs
[train_policy_through_warp.py](../examples/train_policy_through_warp.py) for 300 Adam steps
per approach, discarding the first five timings. Each iteration is synchronized. All four
approaches produced bit-identical parameter gradients on the checked batch and identical
300-step training curves. This is a training measurement, separate from the inference and
callback microbenchmarks above.

Reproduce the Newton cost section directly, or run its full training example; the projectile
script prints its gradient checks, training curves and timing summary:

```bash
python -c "import runpy; print(runpy.run_path('examples/newton_ball_torch.py')['cost_per_iteration']())"
python examples/train_policy_through_warp.py
```
