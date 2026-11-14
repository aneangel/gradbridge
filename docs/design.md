# Design

## Why

Python already standardizes two layers of interoperability:

| Layer | Standard | Status |
|---|---|---|
| Sharing memory | DLPack | supported handoffs, with stream/lifetime constraints |
| Sharing an API | Python Array API | being solved |
| **Sharing gradients** | gradbridge's `DiffFunction` protocol | derivative rules shared through framework adapters |

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
| PyTorch | eager: an `autograd.Function` calling the source directly; transforms: custom ops with fake/vmap rules and an `autograd.Function` JVP; compiled first-order forward mode: explicit primal/tangent custom ops |
| JAX | `jax.custom_vjp` or `custom_jvp` around gradbridge's compiled XLA FFI handler; Warp's `jax_callable` and host callbacks provide fallbacks. JAX sources compose directly. |
| TensorFlow | default: `tf.custom_gradient` around the compiled `GradbridgeCall` op, with a DLPack-aware `tf.py_function` fallback. `native=True`: exportable JAX sources lower through `jax2tf` into a native TensorFlow graph. |
| Warp | `wp.Tape.record_func` |
| NumPy | explicit `.vjp()`, `.jvp()` and scalar `.grad()` methods |

The default JAX wrapper supplies reverse mode. `mode="jvp"` selects forward mode for foreign
functions; JAX cannot transpose the resulting opaque callback to obtain reverse mode.
PyTorch's compiled JVP path unpacks input duals, calls the primal and source JVP ops, and
constructs output duals. This supports first-order JVPs and `jacfwd`; nested compiled JVPs
are explicitly rejected. Eager higher derivatives continue through differentiable rules
when the source supports them.

Eager PyTorch keeps source residuals for reverse gradients by default and skips recording
them during inference. Compiled/transform callback paths, and the JAX/TensorFlow callback
consumers, rebuild the source forward state inside derivative rules. Native graph export
instead exposes forward and backward math to the same compiler for optimization and reuse.

Warp residuals track touched address ranges. A later bridged Warp call can invalidate an
earlier tape; `invalidate_warp_residuals(*arrays)` provides notification for external writes.
Overlapping or unknown footprints trigger recomputation from saved inputs, while independent
residuals remain cached. This does not snapshot hidden mutable inputs.

## Data movement on the GPU

Supported handoffs borrow buffers through DLPack or framework interop. Native callbacks
borrow the consumer's input buffers and copy computed results into its output allocations
on the same device. They avoid host round trips without promising zero device copies.

For a PyTorch source, the compiled JAX/TensorFlow handlers wrap CPU buffers as NumPy/PyTorch
views and CUDA buffers as PyTorch views on the consumer's `ExternalStream`. This supports
arbitrary rank, including scalars and empty arrays, without depending on Warp. Other sources
use Warp's 1–4 dimensional representation in these handlers. Native PyTorch batching in JAX
passes the whole batch and its derivative rules through that path.

The handlers are small C++ libraries compiled and cached on first use. If TensorFlow's handler
is unavailable, its Python callback can still preserve supported GPU allocations through
DLPack. Conversion inspects the allocation's device because callback tensor metadata can
report CPU placement for a GPU allocation. JAX falls back to Warp's `jax_callable` when the
signature permits, then a host callback. Side-effecting VJPs use JAX `io_callback` on the
host path so captured PyTorch parameter-gradient accumulation is not eliminated or repeated.

Handoffs order producer and consumer streams. PyTorch native callbacks run on the consumer's
stream without a host wait; generic Warp-backed callbacks synchronize that stream before
returning to protect result lifetimes. TensorFlow's legacy DLPack interface exports finished
buffers and can also wait on the host. Views may need copies for contiguity or alignment
(XLA: 16 bytes, TensorFlow: 64 bytes). NumPy and unsupported signatures can require host copies;
see [data movement](limitations.md#data-movement) for the fallback contract.

## Performance

Current checkout, measured on 2026-11-13: median ms per forward + backward of
`F(x, y) = sin(x) * y`, RTX 5090, float32, 10M elements
([versions, full tables and reproduction](../benchmarks/RESULTS.md#framework-callback-matrix)):

| Caller ↓ / function written in → | native | torch | JAX | TF | Warp | NumPy |
|---|---|---|---|---|---|---|
| **PyTorch** | 0.346 | 0.348 | 0.650 | 0.691 | 1.192 | 130.896 |
| **JAX** (`jit`) | 0.191 | 0.634 | 0.189 | 0.701 | 0.810 | 79.721 |
| **Warp** | 0.397 | 0.566 | 0.872 | 1.104 | 0.398 | 127.748 |
| **TensorFlow** | 5.823 | 7.059 | 7.541 | 6.592 | 7.459 | 136.410 |

A function composed directly in its own framework needs no bridge. Native and same-framework
columns are separately timed samples and can vary. From PyTorch, the Warp source takes
1.192 ms versus 1.212 ms for the hand-written bridge; JAX takes 0.650 ms versus 1.292 ms.
The TensorFlow row uses eager `GradientTape` and includes `.numpy()` materialization of the
input gradient, so it does not isolate device dispatch overhead. Native graph export has
its own comparison below.

Additional measurements cover separate optimizations and use their own baselines:

- [Downstream inference](downstream.md#inference-performance): skipping backward-state
  recording, 1.09–1.28× on CPU and 1.07–1.20× on the RTX 5090 for the tested workloads.
- [Native JAX callbacks](limitations-audit.md#measured-effect-of-removing-host-fallbacks):
  5.69–40.48× against forced host fallback for scalar, batched and five-dimensional signatures.
- [Native JAX→TensorFlow graph](limitations-audit.md#native-tensorflow-graph-measurement):
  5.68× against a device callback, including whole-computation XLA compilation. The optimized
  GPU program fused JAX math with surrounding TensorFlow operations and gradients.

## Validation

The current recorded CPU/GPU validation passed **505 tests, zero skipped** on an RTX 5090:
Python 3.12.13, PyTorch 2.10.0+cu128, JAX 0.12.0.1, TensorFlow 2.21.0, Warp 1.16.0 and
Newton 1.6.0.dev0. [Downstream validation](downstream.md#tested-environment) records the
full environment, real downstream projects and checks.

CPU CI is configured for Python 3.11 and 3.12, installs downstream dependencies, and treats
unexpected host-copy warnings as errors. Its framework installs are unpinned; the recorded
local environment is a reproducible reference, not a claim that every framework release
combination is supported. Run locally with the required frameworks installed:

```bash
pip install -e ".[all,test,downstream]"
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
    pytest -q -W error::gradbridge.HostCopyWarning
```

## Roadmap

- Multi-GPU validation
- Finer stream handshakes in place of the remaining host waits
- Automatic forward-mode integration for TensorFlow; compiled higher-order derivatives for PyTorch
- Broader native lowering and batching for signatures that still require callbacks or host fallback
- A compatibility matrix to narrow the TensorFlow/Triton import-order guard
- A rollout layer: truncated backpropagation windows, checkpointing and gradient-health reports
  for long differentiable simulations
