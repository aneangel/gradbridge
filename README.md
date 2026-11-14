# gradbridge

[![ci](https://github.com/aneangel/gradbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/aneangel/gradbridge/actions/workflows/ci.yml)

**Gradients that cross framework boundaries.** Wrap a function written in PyTorch, JAX,
TensorFlow, NVIDIA Warp or NumPy, then call it — and differentiate through it — from any of
the others. On the GPU, data stays on the device.

## Getting started

```bash
pip install "gradbridge[torch,jax]"   # pick the frameworks you use: torch, jax, tf, warp
pip install "gradbridge[all]"         # or all four
```

gradbridge itself only requires NumPy and Python 3.10+; each framework adapter is imported
when you first use it.

## Quick example

Call a JAX function from PyTorch, and backpropagate through it:

```python
import gradbridge as gb
import jax.numpy as jnp
import torch

physics = gb.to_torch(gb.from_jax(lambda x: jnp.sin(x) ** 2))   # JAX code...

x = torch.randn(3, requires_grad=True)
physics(x).sum().backward()                                     # ...PyTorch gradients
print(x.grad)                                                   # 2 sin(x) cos(x)
```

Every combination works the same way: a **source** wraps a function
(`from_torch`, `from_jax`, `from_tf`, `from_warp`, `from_numpy`) and a **consumer** exposes it
to another framework (`to_torch`, `to_jax`, `to_tf`, `to_warp`, `to_numpy`). Gradients flow
through `jit`, `torch.compile`, `vmap` and higher-order derivatives where the frameworks allow.

## What it buys you

Training a PyTorch network through a Warp simulation, the usual way is a hand-written
`torch.autograd.Function` around a `wp.Tape`, and for speed, CUDA graphs captured by hand
(fixed buffers, a forward and a backward graph, gradients zeroed at the right moment). With
gradbridge it is one line; `capture=True` captures and manages the graphs for you:

```python
simulate = gb.to_torch(gb.from_warp(rollout, in_dtypes=[wp.vec3], capture=True))
```

Milliseconds per training iteration (forward + backward), each approach without / with CUDA graphs:

| | Native Warp loop | Hand-written bridge | gradbridge |
|---|---|---|---|
| [Newton's `example_diffsim_ball`](https://github.com/aneangel/gradbridge/blob/main/examples/newton_ball_torch.py) (288 substeps) | 62.4 / 5.0 | 62.0 / 5.2 | 63.3 / 5.2 |
| [MLP through a 200-step Warp projectile](https://github.com/aneangel/gradbridge/blob/main/examples/train_policy_through_warp.py) (batch 4096) | — | 15.6 / 2.1 | 15.9 / 2.0 |

Graphs are what make these fast (about 12× on Newton), and gradbridge matches hand-captured
graphs, including Newton's own. What it adds is the code you no longer write and the cases it
keeps correct: several forwards before one backward, `retain_graph`, outputs you hold on to,
new input shapes, buffers shared with other calls. Every approach gives bit-identical gradients
(the Newton example reproduces Newton's own training loop). Measured on one RTX 5090; run the
examples to reproduce. `capture=True` has one rule: every call must launch the same kernels with
the same scalar arguments (see the [guide](https://github.com/aneangel/gradbridge/blob/main/docs/guide.md#many-small-kernels-capturetrue)).

Beyond PyTorch and Warp, the same two calls connect any pair of frameworks, keep GPU data on
the device without copies, and give gradients to network weights used *inside* a bridged
function, not only to its arguments. More examples:

- [Warp simulation optimized with JAX](https://github.com/aneangel/gradbridge/blob/main/examples/warp_sim_in_jax.py)
- [PyTorch policy trained through JAX dynamics and a TensorFlow reward](https://github.com/aneangel/gradbridge/blob/main/examples/mixed_pipeline.py)

## Known limitations

- **Bridge coarse functions.** Each call costs a few tenths of a millisecond and nothing fuses
  across the boundary: wrap a simulator rollout, a model or a loss, not single operations.
- **Inside a JAX hot loop, prefer Warp's own `jax_kernel` for single Warp kernels.** gradbridge
  runs your function in a callback (~0.5 ms per gradient step); `jax_kernel` runs inside XLA.
- **`capture=True` replays captured scalars**: a value that changes between calls (a time, a
  seed) must be passed as an input array. It covers Warp sources only.
- **NumPy functions run on the host**, so their inputs and outputs are copied off the GPU.
- **Calling other frameworks from TensorFlow needs a C++ compiler** for zero-copy; without one,
  gradbridge warns once and falls back to `tf.py_function` (host copies).
- **Second-order derivatives** need a source that supports them: PyTorch, JAX and TensorFlow do;
  Warp is first-order only; NumPy uses finite differences beyond your VJP.
- **Forward mode** is available for the JAX and NumPy consumers only.
- **Warp buffers you write yourself** between forward and backward are invisible to
  gradbridge; pass `cache_residuals=False` to `to_torch` in that case.
- **Import PyTorch (or `gradbridge.torch`) before TensorFlow** if you use `torch.compile`:
  the two bundle conflicting LLVM builds.
- **Tested on one GPU** (NVIDIA RTX 5090) and on CPU; multi-GPU placement is not validated yet.

Details and workarounds: [docs/limitations.md](https://github.com/aneangel/gradbridge/blob/main/docs/limitations.md).

## Learn more

- [User guide](https://github.com/aneangel/gradbridge/blob/main/docs/guide.md): what works with
  each framework, options, NumPy functions, gradient checking, and the `__gradbridge__` protocol
- [Design](https://github.com/aneangel/gradbridge/blob/main/docs/design.md): how it works, how data
  moves between frameworks on the GPU, and performance
- [Benchmarks](https://github.com/aneangel/gradbridge/blob/main/benchmarks/RESULTS.md)

## License

Apache-2.0
