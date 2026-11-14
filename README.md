# gradbridge

[![ci](https://github.com/aneangel/gradbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/aneangel/gradbridge/actions/workflows/ci.yml)

**Gradients that cross framework boundaries.** Wrap a function written in PyTorch, JAX,
TensorFlow, NVIDIA Warp or NumPy, then call it — and differentiate through it — from any of
the others. Supported GPU handoffs share device buffers; NumPy functions execute on the host.

## New in 0.3.0

| Area | What is available now |
|---|---|
| Device handoffs | PyTorch sources called from JAX support native scalar, empty, high-rank and natively batched buffers on CPU/GPU, without requiring Warp. TensorFlow's native PyTorch path also supports arbitrary tensor rank. |
| TensorFlow without a compiler | Supported GPU buffers pass through the DLPack fallback without forced NumPy copies. The compiled op remains the default fast path. |
| Cross-framework fusion | `gb.to_tf(jax_source, native=True)` exports compatible JAX functions into a TensorFlow graph for XLA compilation with surrounding operations. |
| Forward derivatives | Eager PyTorch JVPs, `jacfwd` and low-level forward AD; first-order JVPs and `jacfwd` compiled with `aot_eager` or Inductor; explicit JVP evaluation in TensorFlow and Warp. |
| Inference | Eager PyTorch evaluation skips recording source backward state when gradients are not needed. |
| Warp buffers and placement | Selective residual invalidation after external writes, plus a fix for conversions that ignored an explicit destination device. CPU↔CUDA placement is verified. |

The [limitation audit](https://github.com/aneangel/gradbridge/blob/main/docs/limitations-audit.md) records each fix, the GPU measurements and
viable paths for restrictions that still need compiler integration or additional hardware.

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

Milliseconds per iteration on the current checkout, each approach without / with CUDA graphs.
Newton measures forward + backward; the policy workload also includes the Adam update:

| | Native Warp loop | Hand-written bridge | gradbridge |
|---|---|---|---|
| [Newton's `example_diffsim_ball`](https://github.com/aneangel/gradbridge/blob/main/examples/newton_ball_torch.py) (288 substeps) | 67.8 / 4.0 | 70.1 / 4.1 | 69.2 / 4.2 |
| [MLP through a 200-step Warp projectile](https://github.com/aneangel/gradbridge/blob/main/examples/train_policy_through_warp.py) (batch 4096) | — | 15.5 / 1.9 | 15.2 / 1.8 |

Graphs are what make these fast (about 17× on Newton), and gradbridge matches hand-captured
graphs, including Newton's own. What it adds is the code you no longer write and the cases it
keeps correct: several forwards before one backward, `retain_graph`, outputs you hold on to,
new input shapes, buffers shared with other calls. Every approach gives bit-identical gradients
(the Newton example reproduces Newton's own training loop). Measured on one RTX 5090; see the
[current results and methodology](https://github.com/aneangel/gradbridge/blob/main/benchmarks/RESULTS.md#warp-simulation-and-policy-workloads)
to reproduce. `capture=True` has one rule: every call must launch the same kernels with
the same scalar arguments (see the [guide](https://github.com/aneangel/gradbridge/blob/main/docs/guide.md#many-small-kernels-capturetrue)).

Beyond PyTorch and Warp, the same two calls connect any pair of frameworks, share supported
GPU buffers, and give gradients to network weights used *inside* a bridged function, not only
to its arguments. See the [remaining host fallbacks](https://github.com/aneangel/gradbridge/blob/main/docs/limitations.md#data-movement).
More examples:

- [Warp simulation optimized with JAX](https://github.com/aneangel/gradbridge/blob/main/examples/warp_sim_in_jax.py)
- [PyTorch policy trained through JAX dynamics and a TensorFlow reward](https://github.com/aneangel/gradbridge/blob/main/examples/mixed_pipeline.py)

## Forward-mode sensitivities

PyTorch callers can use `torch.func.jvp`, `torch.func.jacfwd` and
`torch.autograd.forward_ad` through the bridge. For the JAX `physics` function above:

```python
direction = torch.ones_like(x)
value, sensitivity = torch.func.jvp(physics, (x,), (direction,))
jacobian = torch.func.jacfwd(physics)(x)
```

The source supplies the JVP rule. These eager transforms compose with `vmap` and with reverse
mode, including Hessians, when the source can differentiate its derivative rules. JAX and
NumPy consumers also support forward mode. NumPy sources approximate JVPs with directional
finite differences unless you provide `jvp=`; Warp sources have no forward-mode rule.

First-order JVPs and forward Jacobians also work with `torch.compile`; compile the complete
transform:

```python
compiled_jvp = torch.compile(lambda x, v: torch.func.jvp(physics, (x,), (v,)))
compiled_jacobian = torch.compile(torch.func.jacfwd(physics))
```

These paths are tested with `aot_eager` and Inductor. Nested compiled JVPs raise an error.
TensorFlow and Warp callers can evaluate a
supplied JVP explicitly with `gb.to_tf(dfn.jvp_function)` or `gb.to_warp(dfn.jvp_function)`.

## Faster inference

In eager PyTorch, `torch.no_grad()`, `torch.inference_mode()` and calls without any tensor
input requiring gradients skip building the source's backward graph or tape. This avoids
recording backward state during evaluation. Forward-mode AD still works under `no_grad()`.

Measured inference speedup against explicitly recording source backward state under `no_grad()`:

| Downstream workload | CPU (Ryzen 7 7800X3D) | GPU (RTX 5090) |
|---|---:|---:|
| torchdiffeq: 20-step RK4 solve | 1.28× | 1.12× |
| Diffrax: 20-step Tsit5 solve | 1.26× | 1.08× |
| Keras: three-layer MLP, batch 32 | 1.09× | 1.07× |
| Newton: contact rollout, 288 substeps | 1.10× | 1.20× |

Both paths use identical inputs and checked outputs. Results are median per-call latencies
over seven alternating rounds of 20 calls after warmup; each CUDA call is synchronized.
See [versions, timings and methodology](https://github.com/aneangel/gradbridge/blob/main/docs/downstream.md#inference-performance), or reproduce
from a source checkout with the downstream dependencies installed:

```bash
python benchmarks/bench_inference.py
python benchmarks/bench_inference.py --device cuda
```

## Faster device handoffs

JAX calls to PyTorch now keep scalar, empty, high-rank and natively batched arrays on their
device. For a value and gradient over 1,048,576 float32 elements on the RTX 5090:

| Signature | Forced host fallback (ms) | Native GPU callback (ms) | Speedup |
|---|---:|---:|---:|
| Scalar loss | 2.008 | 0.353 | 5.69× |
| Batched input | 6.672 | 0.340 | 19.61× |
| Five-dimensional input | 14.101 | 0.348 | 40.48× |

Both paths use the current derivative rules; the baseline forces host fallback for the same
signatures. Outputs and gradients are checked before timing, warmup is excluded,
and results are medians of seven alternating rounds of ten synchronized calls. These are
measurements of the tested signatures, not a universal model speedup.

```bash
python benchmarks/bench_jax_native.py --markdown native-callbacks.md
```

TensorFlow can also exchange supported GPU buffers without a compiler: the fallback uses
DLPack instead of NumPy copies. The compiled op remains the default fast path.

## Native graph lowering

For exportable JAX functions, `native=True` converts the function into a TensorFlow graph
that can compile with the surrounding computation:

```python
import gradbridge as gb
import jax.numpy as jnp
import tensorflow as tf

jax_physics = gb.from_jax(lambda x: jnp.sin(x) ** 2)
tf_physics = gb.to_tf(jax_physics, native=True)
compiled_tf = tf.function(tf_physics, jit_compile=True, autograph=False)
```

This path supports first and second derivatives on CPU/GPU and allows XLA to fuse JAX math
with TensorFlow operations. A tested 1M-element value-and-gradient computation measured
**5.68× faster** than the device callback. It requires compatible JAX/TensorFlow versions
and an exportable function; export errors are reported without silently switching to a
callback. The default callback path remains available for functions that call other foreign
frameworks. See the [audit and tested versions](https://github.com/aneangel/gradbridge/blob/main/docs/limitations-audit.md#native-tensorflow-graph-measurement),
or reproduce the measurement and save the optimized GPU program:

```bash
python benchmarks/bench_tf_native.py --markdown native-tf.md --hlo optimized.hlo
```

## External Warp buffer writes

When external code overwrites a Warp intermediate between forward and backward, call
`gb.invalidate_warp_residuals(buffer)` after the write. It invalidates only tapes whose memory
overlaps that array, with no data copies or GPU synchronization. Passing no arrays invalidates
all cached Warp tapes. Unaffected tapes retain their cached state.

Backward recomputes invalidated forwards, so they must rebuild intermediates from their saved
inputs. Pass changing model inputs explicitly; notification does not snapshot hidden state.
If writes cannot be annotated, `gb.to_torch(dfn, cache_residuals=False)` always recomputes
the forward under the same reconstruction requirement.

## Validation

**505 tests passed, zero skipped**, across CPU and the RTX 5090, with unexpected host-copy
warnings treated as errors. Coverage includes reverse mode, forward mode, mixed derivatives,
batching, compiled reverse/forward mode and CUDA graph replay. The limitation audit added
**71 regression checks**, including GPU pointer identity, native TensorFlow XLA export and
derivatives, selective Warp invalidation and changing array controls during graph replay.

Real downstream integrations exercise **SciPy** (NumPy), **torchdiffeq** (PyTorch), **Diffrax**
(JAX), **Keras** (TensorFlow) and **Newton** (Warp). Tests compare bridged values and derivatives
with native or analytic references and check device placement. SciPy computation stays on the
host; its GPU integration returns results and gradients to CUDA tensors.

See the [downstream tests, tested versions and reproduction steps](https://github.com/aneangel/gradbridge/blob/main/docs/downstream.md).
From a source checkout with CUDA-enabled frameworks installed, run the full suite with:

```bash
pip install -e ".[all,test,downstream]"
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
    pytest -q -W error::gradbridge.HostCopyWarning
```

## Known limitations

- **Bridge coarse functions on callback paths.** Dispatch costs time and callback boundaries
  prevent fusion. Wrap a rollout, model or loss; use native graph lowering for exportable JAX
  functions in TensorFlow to remove that boundary.
- **Inside a JAX hot loop, prefer Warp's own `jax_kernel` for single Warp kernels.** gradbridge
  runs your function in a callback; `jax_kernel` lowers the kernel directly inside XLA.
- **`capture=True` replays captured scalars**: a value that changes between calls (a time, a
  seed) must be passed as an input array. It covers Warp sources only.
- **NumPy functions run on the host**, so their inputs and outputs are copied off the GPU.
- **Second-order derivatives** need a source that supports them: PyTorch, JAX and TensorFlow do;
  Warp is first-order only; NumPy uses finite differences beyond your VJP.
- **Forward mode needs a source JVP rule.** PyTorch, JAX and NumPy consumers support it;
  TensorFlow and Warp can evaluate the rule explicitly. Automatic TensorFlow
  `ForwardAccumulator` integration is not supported; Warp sources have no JVP rule.
- **Compiled higher derivatives:** first-order compiled JVPs and `jacfwd` work; nested compiled
  JVPs and compiled double backward are not supported. Eager higher derivatives remain available.
- **External Warp buffer writes need notification.** Call `gb.invalidate_warp_residuals(buffer)`
  after your own writes and before backward, or disable residual caching. Unknown memory
  footprints are invalidated conservatively. See the
  [reconstruction contract](#external-warp-buffer-writes) above.
- **Import order can depend on binary builds.** Import PyTorch (or `gradbridge.torch`) before
  TensorFlow to avoid the conservative guard for potentially conflicting LLVM libraries.
- **Single-GPU validation:** CPU and NVIDIA RTX 5090 are validated; multi-GPU placement is
  not validated yet.

Details and workarounds: [docs/limitations.md](https://github.com/aneangel/gradbridge/blob/main/docs/limitations.md).
The [limitation audit](https://github.com/aneangel/gradbridge/blob/main/docs/limitations-audit.md) records current support and measurements,
and identifies viable paths for the remaining compiler and hardware work.

## Learn more

- [User guide](https://github.com/aneangel/gradbridge/blob/main/docs/guide.md): what works with
  each framework, options, NumPy functions, gradient checking, and the `__gradbridge__` protocol
- [Design](https://github.com/aneangel/gradbridge/blob/main/docs/design.md): how it works, how data
  moves between frameworks on the GPU, and performance
- [Benchmarks](https://github.com/aneangel/gradbridge/blob/main/benchmarks/RESULTS.md)
- [Downstream validation](https://github.com/aneangel/gradbridge/blob/main/docs/downstream.md): CPU/GPU integration tests and inference benchmarks

## License

Apache-2.0
