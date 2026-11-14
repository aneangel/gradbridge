# Design

## Why

Python already standardizes two layers of interoperability:

| Layer | Standard | Status |
|---|---|---|
| Sharing memory | DLPack | solved |
| Sharing an API | Python Array API | being solved |
| **Sharing gradients** | — | **nothing** |

DLPack does not carry gradient information, so every pair of frameworks needs its own
hand-written bridge (`jax2torch`, `torch2jax`, per-project `torch.autograd.Function` wrappers
around Warp kernels, ...). Most cover one direction, first-order gradients and eager mode only.
gradbridge defines one small protocol for differentiable functions and ships an adapter for each
framework: **N adapters instead of N² bridges.**

## How it works

A `DiffFunction` maps a tuple of arrays to a tuple of arrays and produces its derivative rules
**as further DiffFunctions**:

- `vjp_function`: `g(*xs, *cotangents) -> input cotangents`
- `jvp_function`: `h(*xs, *tangents) -> output tangents`

Because the rules are themselves DiffFunctions, a consumer that knows how to call a DiffFunction
also knows how to differentiate it again, as far as the source allows. Consumers hook into each
framework's own extension point:

| Framework | Mechanism |
|---|---|
| PyTorch | eager: an `autograd.Function` calling the source directly; under `torch.compile`/`torch.func`: a `torch.library.custom_op` (fake kernel, vmap rule) inside an `autograd.Function` |
| JAX | `jax.custom_vjp` (or `custom_jvp`) around an XLA FFI call (Warp's `jax_callable`), or `jax.pure_callback` when Warp is not installed; JAX sources are composed directly |
| TensorFlow | `tf.custom_gradient` around gradbridge's compiled `GradbridgeCall` op (TensorFlow's own buffers and stream), with `tf.py_function` as fallback |
| Warp | `wp.Tape.record_func` |

## Data movement on the GPU

Arrays are shared zero-copy through DLPack or the frameworks' own interop functions. Every
handoff orders the producer's CUDA stream before the consumer's; where CUDA already orders them
(PyTorch's legacy default stream with a blocking stream), the explicit wait is skipped. Views that
start mid-allocation are realigned for the frameworks that require it (XLA: 16 bytes,
TensorFlow: 64 bytes).

## Performance

Median ms per forward+backward of an elementwise function, RTX 5090, float32, 10M elements
([full tables and method](../benchmarks/RESULTS.md), `python benchmarks/bench.py`):

| Caller ↓ / function written in → | native | torch | JAX | TF | Warp | NumPy |
|---|---|---|---|---|---|---|
| **PyTorch** | 0.34 | 0.35 | 0.61 | 0.68 | 1.27 | 126 |
| **JAX** (`jit`) | 0.19 | 0.67 | 0.20 | 0.71 | 0.85 | 81 |
| **Warp** | 0.40 | 0.66 | 0.85 | 1.12 | 0.40 | 126 |
| **TensorFlow** | 6.6 | 7.2 | 7.1 | 6.4 | 7.3 | 133 |

A function called from its own framework runs at native speed (the diagonal). Against bridges
written by hand, from PyTorch: Warp 1.27 ms vs 1.27 ms, JAX 0.61 ms vs 1.29 ms. For a whole
simulator the bridge is within run-to-run noise (about 5%): one optimization step through a
Newton rollout took 62–64 ms natively, 62–66 ms with a hand-written `autograd.Function`, and
65–67 ms through gradbridge.

## Tested versions

- CI (CPU): Python 3.11 and 3.12 with current PyTorch, JAX, TensorFlow and Warp releases.
- GPU (RTX 5090): Python 3.12, PyTorch 2.11 (cu128), JAX 0.12 (cuda12), TensorFlow 2.21
  (and-cuda), Warp 1.19, Newton 1.7.0.dev0.

Run the tests with:

```bash
pip install -e ".[all,test]"
pytest
```

## Roadmap

- Multi-GPU validation
- Finer stream handshakes in place of the remaining host waits
- Forward mode for the PyTorch and TensorFlow consumers
- A rollout layer: truncated backpropagation windows, checkpointing and gradient-health reports
  for long differentiable simulations
