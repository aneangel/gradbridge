# gradbridge

**Gradients that cross framework boundaries.** Wrap a function from PyTorch, JAX,
TensorFlow, NVIDIA Warp or NumPy, then call it — and differentiate through it —
from any of the others.

```python
import gradbridge as gb
import jax.numpy as jnp
import torch

physics = gb.to_torch(gb.from_jax(lambda x: jnp.sin(x) ** 2))   # JAX code...

x = torch.randn(3, requires_grad=True)
physics(x).sum().backward()                                     # ...PyTorch gradients
print(x.grad)                                                   # 2 sin(x) cos(x)
```

## Why

Python already standardizes two layers of interoperability:

| Layer | Standard | Status |
|---|---|---|
| Sharing memory | DLPack | solved |
| Sharing an API | Python Array API | being solved |
| **Sharing gradients** | — | **nothing** |

DLPack does not carry gradient information, so every pair of frameworks needs its
own hand-written bridge (`jax2torch`, `torch2jax`, per-project `torch.autograd.Function`
wrappers around Warp kernels, ...). Most of them cover one direction, first-order
gradients, and eager mode only.

gradbridge defines one small protocol for differentiable functions and ships an
adapter for each framework: **N adapters instead of N² bridges.**

## Install

```bash
pip install -e ".[all]"        # or pick: .[torch] .[jax] .[tf] .[warp]
```

Only NumPy is required; framework adapters are imported lazily.

## Usage

**Sources** turn a framework function into a protocol object:
`from_torch`, `from_jax`, `from_tf`, `from_numpy`, `from_warp`.

**Consumers** expose a protocol object to a framework:
`to_torch`, `to_jax`, `to_tf`, `to_numpy`, `to_warp`.

```python
f = gb.from_tf(tf_fn)            # any source
g = gb.to_jax(f)                 # any consumer
jax.jit(jax.grad(lambda x: g(x).sum()))(x)

# or in one step
g = gb.convert(tf_fn, src="tf", dst="jax")
```

### A Warp simulation, optimized with JAX

```python
@wp.kernel
def step(pos_in: wp.array(dtype=wp.vec2d), vel_in: ..., pos_out: ..., vel_out: ...):
    ...

def simulate(v0):                       # launches 200 Warp kernel steps
    ...
    return final_position

sim = gb.to_jax(gb.from_warp(simulate, in_dtypes=[wp.vec2d]))
grad = jax.jit(jax.grad(lambda v: jnp.sum((sim(v.reshape(1, 2))[0] - target) ** 2)))
```

See `examples/warp_sim_in_jax.py` (converges to the target in 60 steps) and
`examples/mixed_pipeline.py` (a PyTorch policy trained through JAX dynamics and a
TensorFlow reward with one `loss.backward()`).

### Train through a Newton simulation from PyTorch

```python
simulate = gb.to_torch(gb.from_warp(scene.rollout, in_dtypes=[wp.vec3]))  # Newton rollout
v0 = torch.tensor([[0.0, 5.0, -5.0]], device="cuda", requires_grad=True)
loss = ((simulate(v0) - target) ** 2).sum()
loss.backward()                                  # gradients through contacts, on the GPU
```

[examples/newton_ball_torch.py](examples/newton_ball_torch.py) runs Newton's own
`example_diffsim_ball` scene from `torch.optim.SGD` and reproduces Newton's native training
loop bit for bit, then trains a PyTorch MLP policy through the simulation.

### NumPy functions

NumPy has no autodiff, so give a VJP or fall back to finite differences (with a warning):

```python
f = gb.from_numpy(lambda x: x ** 3, vjp=lambda x, ct: (3 * x ** 2 * ct,))
```

### Join the protocol from your own library

Any object with a `__gradbridge__()` method returning a `gb.DiffFunction` works with
every consumer, like `__dlpack__` does for memory:

```python
class MySimulator:
    def __gradbridge__(self):
        return gb.from_warp(self._step)

torch_sim = gb.to_torch(MySimulator())
```

### Check gradients the same way everywhere

```python
gb.gradcheck(f, x, y)             # analytic VJP vs finite differences
gb.gradcheck(f, x, y, order=2)    # VJP-of-VJP (Hessian-vector products)
gb.compare(torch_impl, jax_impl, x, y)   # do two implementations agree?
```

```
gradcheck 'F_numpy' (order 1): PASSED
  input  shape           max abs err  max rel err  ok
      0  (3,)              2.943e-11    6.504e-11  yes
      1  (3,)              5.374e-12    4.717e-11  yes
```

## What works

Tested on CPU and on an NVIDIA GPU (RTX 5090): every source × every consumer, in float32 and
float64, plus the transformations below. On GPU the tests also assert that results stay on the
device, keep their dtype, and (for the zero-copy paths) that no device↔host copy happens.

| Consumer | First-order | Compiled | Batched | Second-order | Forward mode |
|---|---|---|---|---|---|
| **PyTorch** | `.backward()`, `torch.autograd.grad` | `torch.compile(fullgraph=True)` (aot_eager, inductor) | `torch.func.vmap`, `vmap(grad(...))` | double backward (`create_graph=True`) | — |
| **JAX** | `jax.grad`, `jax.vjp`, `jacrev` | `jax.jit` | `jax.vmap` | `grad(grad(...))`, also under `jit` | `to_jax(f, mode="jvp")`: `jax.jvp`, `jacfwd` |
| **TensorFlow** | `GradientTape` | `tf.function` (static shapes) | `tf.map_fn` | nested tapes | — |
| **NumPy** | `.vjp()`, `.grad()` | — | — | via `vjp_function` | `.jvp()` |
| **Warp** | `wp.Tape` + `tape.backward()` | — | — | — | — |

Second-order availability depends on the **source**:

| Source | Max order | How |
|---|---|---|
| PyTorch | any | `torch.autograd.grad(create_graph=True)` |
| JAX | any | `jax.vjp` of `jax.vjp` |
| TensorFlow | any | nested `GradientTape` |
| NumPy | 1 exact (your `vjp=`), higher by finite differences | or pass a `DiffFunction` as `vjp=` |
| Warp | 1 | Warp's tape is first-order; order 2 raises `gb.MaxOrderError` with an explanation |

Other options:
- A function called from its own framework (PyTorch from PyTorch, JAX from JAX, TensorFlow
  from TensorFlow, Warp from Warp) is composed natively: no bridge, native speed.
- Tensors a PyTorch function captures (e.g. an `nn.Module`'s parameters) receive gradients
  whichever framework drives the call, accumulated into `.grad` like `loss.backward()` — so a
  PyTorch network inside a Warp, JAX or TensorFlow computation trains through it.
- `gb.to_torch(f, cache_residuals=False)` recomputes the forward inside backward instead of
  keeping the source's residuals (the default keeps them; see Limitations for Warp).
- `from_*(..., batched=True)` declares that a function accepts a leading batch dimension,
  so `vmap` makes one call instead of looping.
- `from_warp(..., in_dtypes=[wp.vec3d])` maps `(N, 3)` arrays to Warp vector arrays.

## Limitations (read before relying on it)

- **Data movement on GPU.** Arrays cross between PyTorch, JAX, TensorFlow and Warp
  zero-copy, with each pair's CUDA streams ordered explicitly. The exceptions, which copy
  device→host→device: NumPy functions (by nature), and calls the device paths cannot
  express — natively `batched=True` sources under JAX, 0-d or >4-d arrays, dtypes Warp has
  no scalar type for. An unexpected fallback emits `gb.HostCopyWarning`; turn it into an
  error with `warnings.simplefilter("error", gb.HostCopyWarning)` to catch regressions.
- **Calling other frameworks from TensorFlow needs a C++ compiler for zero-copy.**
  `tf.py_function` is a host op (TensorFlow places its inputs on the CPU even under
  `tf.device("/GPU:0")`), so gradbridge ships a small TensorFlow op (`_tf_op.cc`) that hands
  the bridged function TensorFlow's own GPU buffers. It is compiled on first use against your
  TensorFlow and cached in `~/.cache/gradbridge` (`CXX` and `GRADBRIDGE_CACHE_DIR` override the
  compiler and location). The package itself stays pure Python: without a compiler, gradbridge
  warns once and falls back to `tf.py_function`.
- **Some zero-copy handoffs synchronize the host** instead of copying: TensorFlow only
  exports finished tensors and imports without a stream handshake, and the JAX and
  TensorFlow consumer callbacks wait for their framework's stream before returning (measured:
  no visible cost). No copies, but a host wait per call.
- **Overhead.** A few tenths of a millisecond per call for small arrays (see
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md)); nothing fuses across the boundary.
  Bridge coarse functions (a simulator rollout, a model, a loss), not elementwise ops.
- **Warp buffers written outside the bridge.** In eager PyTorch the forward's residuals
  (for Warp: the tape) are kept for backward. gradbridge tracks the memory each Warp
  forward touched and recomputes any residual a later bridged Warp call may have
  overwritten (e.g. several rollouts of one simulator before one `backward()`), so those
  gradients stay exact. It cannot see *your own* code writing into those buffers between
  forward and backward; pass `cache_residuals=False` to always recompute if you do that.
  Under `torch.compile`/`torch.func` the forward is always recomputed.
- **TensorFlow:** no XLA (`jit_compile=True`), and input shapes must be static inside
  `tf.function`. `tf.vectorized_map` gives correct values but TensorFlow cannot
  backpropagate through its loop fallback (reproducible without gradbridge) — use
  `tf.map_fn` for gradients.
- **Forward mode** is available for the JAX and NumPy consumers only.
- **torch.compile + double backward** together is not supported (the compiled backward
  calls an opaque VJP op). Each works separately.
- **Single GPU tested.** Multi-device placement has not been validated.
- **Import order with TensorFlow:** TensorFlow and Triton (used by `torch.compile`)
  bundle conflicting LLVM builds. Import `gradbridge.torch` before TensorFlow; gradbridge
  raises a clear `ImportError` rather than letting the process abort.

## Performance

Median ms per forward+backward of an elementwise function on an RTX 5090, float32,
10M elements ([full tables and method](benchmarks/RESULTS.md), `python benchmarks/bench.py`):

| Caller ↓ / function written in → | native | torch | JAX | TF | Warp | NumPy |
|---|---|---|---|---|---|---|
| **PyTorch** | 0.34 | 0.35 | 0.61 | 0.68 | 1.27 | 126 |
| **JAX** (`jit`) | 0.19 | 0.67 | 0.20 | 0.71 | 0.85 | 81 |
| **Warp** | 0.40 | 0.66 | 0.85 | 1.12 | 0.40 | 126 |
| **TensorFlow** | 6.6 | 7.2 | 7.1 | 6.4 | 7.3 | 133 |

Everything except the NumPy column stays on the GPU. A function called from its own framework
runs at native speed (the diagonal). Against the bridges people write by hand, from PyTorch:
Warp 1.27 ms vs 1.27 ms hand-written, JAX 0.61 ms vs 1.29 ms. For a whole simulator the bridge
is within run-to-run noise (about 5%): across runs, one optimization step through a Newton
rollout took 62–64 ms natively, 62–66 ms with a hand-written `autograd.Function`, and 65–67 ms
through gradbridge's default, which also guards against simulator buffer reuse.

## How it works

A `DiffFunction` maps a tuple of arrays to a tuple of arrays and produces its
derivative rules **as further DiffFunctions**:

- `vjp_function`: `g(*xs, *cotangents) -> input cotangents`
- `jvp_function`: `h(*xs, *tangents) -> output tangents`

Because rules are themselves DiffFunctions, a consumer that knows how to call a
DiffFunction automatically knows how to differentiate it again, as far as the
source allows. Consumers hook into each framework's native extension point:

| Framework | Mechanism |
|---|---|
| PyTorch | eager: an `autograd.Function` calling the source directly; under `torch.compile`/`torch.func`: `torch.library.custom_op` (fake kernel, vmap rule) inside an `autograd.Function` with `setup_context` and a generated vmap rule |
| JAX | `jax.custom_vjp` (or `custom_jvp`) around an XLA FFI call on CUDA (Warp's `jax_callable`), `jax.pure_callback` elsewhere; JAX sources are composed directly |
| TensorFlow | `tf.custom_gradient` around gradbridge's compiled `GradbridgeCall` op (TensorFlow's own buffers and stream), `tf.py_function` as fallback |
| Warp | `wp.Tape.record_func` |

## Tested versions

CPU: Python 3.13, PyTorch 2.15, JAX 0.12, TensorFlow 2.21, Warp 1.19, NumPy 2.5.
GPU (RTX 5090, CUDA 12.8/13): Python 3.12, PyTorch 2.11 (cu128), JAX 0.12 (cuda12),
TensorFlow 2.21 (and-cuda), Warp 1.19, Newton 1.7.0.dev0.

```bash
pip install -e ".[all,test]"
pytest
```

## Roadmap

- Multi-device validation
- Finer stream handshakes in place of the remaining host waits
- Forward mode for PyTorch and TensorFlow consumers
- A rollout layer on top: truncated backprop windows, checkpointing and gradient health
  reports for long differentiable simulations

## License

Apache-2.0
