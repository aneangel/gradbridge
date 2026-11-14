# Known limitations

See the [limitation audit](limitations-audit.md) for tested fixes, measured improvements and
viable implementation paths for each remaining restriction.

## Performance

- **Per-call overhead.** A few tenths of a millisecond for small arrays (see the
  [benchmarks](../benchmarks/RESULTS.md)) on callback paths, which prevent fusion. Bridge coarse
  functions — a simulator rollout, a model, a loss. For exportable JAX functions,
  `to_tf(..., native=True)` removes the callback and allows TensorFlow XLA fusion.
- **Recomputation under compilers.** In eager PyTorch the forward's residuals are kept for the
  backward pass. Under `torch.compile` and `torch.func`, and in the JAX and TensorFlow
  callback paths, the backward pass re-runs the forward to rebuild them. Native graph lowering
  exposes the forward and backward to the same optimizer, which can reuse intermediates.
- **Some zero-copy handoffs wait on the host** instead of copying: TensorFlow only exports
  finished tensors through its legacy DLPack interface. Generic Warp-backed JAX and TensorFlow
  callbacks also wait for the consumer stream before returning to protect result lifetimes.
  Native PyTorch callbacks run on the consumer's CUDA stream without that host wait.
- **Warp kernels called from JAX in a hot loop.** gradbridge calls your Warp *function* (with
  its tape) from an XLA callback. Warp's own `wp.jax_kernel` launches a single kernel directly
  from inside XLA and avoids that Python callback. When the work is one
  Warp kernel whose outputs have the shape of its launch grid, use `jax_kernel` for the JAX side;
  gradbridge is the better fit for whole functions, other source frameworks, and PyTorch callers.
  For functions with many kernels, `from_warp(..., capture=True)` removes most of the per-launch
  cost ([guide](guide.md#many-small-kernels-capturetrue)).

## CUDA graphs

- **`from_warp(..., capture=True)` replays captured scalars.** See the contract in the
  [guide](guide.md#many-small-kernels-capturetrue): a scalar argument that changes between
  calls is replayed with its captured value without any error. Use array-valued controls
  for changing values; captured values and gradients with changing input arrays are GPU tested.
- **Graphs cover Warp sources only.** A Warp program that calls a PyTorch network inside (via
  `to_warp`) records Python callbacks on its tape and runs without graphs.
- **Outer `torch.cuda.graph` capture is unsupported for foreign bridged calls**, with or
  without `capture=True`. Arbitrary callbacks, allocations and cross-stream ordering do not
  have a general external-capture protocol. `capture=True` is the tested way to get graph
  replays for the Warp source; specialized outer capture remains a possible integration.

## Data movement

Supported handoffs borrow device buffers and order producer/consumer CUDA streams. Native
callbacks still copy computed results into the consumer's output allocation on the same
device. Alignment, contiguity, dtype or explicit device changes may also require copies.
These paths can instead require a device→host→device round trip:

- NumPy functions (by nature).
- Under the JAX consumer: non-PyTorch `batched=True` sources, non-PyTorch 0-d or >4-d arrays,
  and unsupported native dtypes. PyTorch sources use native buffers for arbitrary rank and
  batching when the compiled handler is available.
- JAX foreign callbacks without a compiled handler or a compatible Warp fallback. A PyTorch
  source needs no Warp when the compiled handler is available; other sources use Warp for
  device callbacks. JAX's final host fallback is `pure_callback`, or `io_callback` for
  side-effecting VJPs.

TensorFlow's compiler-free Python callback now preserves supported GPU allocations through
DLPack. A C++ compiler is needed for its faster native handler, not for GPU buffer sharing.
The JAX→TensorFlow fallback also works without Warp. Other unsupported DLPack signatures may
still fall back to host conversions.

The compiled JAX/TensorFlow dtype set is `bool`, `int8`, `uint8`, `int32`, `int64`, `float16`,
`float32` and `float64`; the non-PyTorch handlers additionally require 1–4 dimensional arrays.
This is a handler contract, not a promise that every source supports every dtype or signature.

Unexpected device→host conversion emits `gb.HostCopyWarning`. Turn it into an error with
`warnings.simplefilter("error", gb.HostCopyWarning)` to catch regressions.
JAX native-handler build failure also emits that warning before trying fallback paths,
even if a compatible Warp device path is available. TensorFlow native-handler build failure
uses `RuntimeWarning`, because its DLPack fallback need not copy to the host.

## TensorFlow

- **The native fast path uses a C++ compiler.** The small TensorFlow op (`_tf_op.cc`) passes
  TensorFlow's buffers and stream directly to the source. It is compiled on first use and
  cached in `~/.cache/gradbridge`; `CXX` and `GRADBRIDGE_CACHE_DIR` override the compiler and
  location. Without it, a one-time `RuntimeWarning` reports the slower Python callback.
  That callback preserves supported GPU buffers through DLPack, even when TensorFlow reports
  CPU placement for those allocations. Unsupported DLPack conversions can still copy through
  the host and emit `HostCopyWarning`.
- The callback path has no XLA (`jit_compile=True`) lowering and needs static input shapes
  inside `tf.function`. `to_tf(jax_source, native=True)` supports XLA for exportable functions
  and compatible JAX/TensorFlow versions. It does not silently fall back when export fails.
- On the callback path, `tf.vectorized_map` gives correct values but gradients through its
  loop fallback fail in the tested configuration (also reproducible without gradbridge).
  Use the tested `tf.map_fn` path for gradients. This is not a blanket restriction on native
  TensorFlow functions or every graph exported from JAX.
- **Import order is binary/version dependent.** Some TensorFlow/Triton combinations have
  conflicting LLVM builds. The protective import guard is conservative: a TensorFlow-first
  CUDA compilation succeeded with the installed PyTorch 2.10.0/TensorFlow 2.21.0 binaries.
  Until the guard is narrowed against a compatibility matrix, import `gradbridge.torch`
  before TensorFlow.

## Derivatives

- **Second order** needs a source that supports it: PyTorch, JAX and TensorFlow go to any order;
  Warp's tape is first-order (order 2 raises `gb.MaxOrderError`); NumPy uses finite differences
  beyond a plain callable VJP. Source operations and their own custom rules can impose
  additional restrictions. A differentiable `DiffFunction` supplied as a custom derivative
  rule can provide exact higher orders; that does not add higher-order AD to Warp's tape.
- **Forward mode** is available for the PyTorch, JAX and NumPy consumers when the source
  supplies a JVP rule (Warp does not; NumPy falls back to directional finite differences).
  PyTorch supports `torch.func.jvp`, `jacfwd` and low-level `forward_ad` eagerly, including
  `no_grad()`. First-order JVPs and `jacfwd` can be compiled when the whole transform is inside
  `torch.compile`; compiled nested JVPs are rejected. TensorFlow and Warp consumers can call
  `dfn.jvp_function` explicitly through `to_tf` / `to_warp`. Automatic TensorFlow
  `ForwardAccumulator` integration for foreign callbacks remains unsupported. The native
  JAX→TensorFlow graph path is separately tested for first and second reverse derivatives.
- **`torch.compile` together with double backward** is not supported (the compiled backward
  calls an opaque VJP op, and native AOTAutograd also rejects double backward in the tested
  version). Each works on its own. Compile the complete first-order JVP/`jacfwd` transform;
  wrapping an already compiled function in an eager JVP is not the supported composition.

## Warp buffers

In eager PyTorch, a Warp source's tape is kept for the backward pass. gradbridge records the
memory each Warp forward touched and recomputes any tape that a later bridged Warp call may
have overwritten (for example several rollouts of one simulator before one `backward()`), so
those gradients stay exact. For *your own* writes between forward and backward, call
`gb.invalidate_warp_residuals(*arrays)` after writing and before backward. Only overlapping
tapes are invalidated; notification performs no copies or GPU synchronization. With no arrays,
it invalidates all cached Warp tapes. Unknown memory footprints are treated conservatively
as overlapping. Notification does not order external CUDA streams; the caller must ensure
the write is properly ordered. Pass `cache_residuals=False` to `to_torch` to always
recompute when those writes cannot be annotated. Recomputation must rebuild intermediates
from the original saved inputs; pass changing model inputs explicitly instead of changing
hidden inputs between forward and backward. Unreported writes remain invisible.

## Platforms

Tested on Linux, on CPU and on one NVIDIA GPU (RTX 5090). Multi-GPU placement has not been
validated yet. Explicit Warp-array CPU↔CUDA conversion now honors the requested destination;
that check does not establish peer-copy or stream correctness across two physical GPUs.

The [tested environment](downstream.md#tested-environment) records versions for the 505-test
CPU/GPU run. Minimum dependency versions in packaging and unpinned CI installs do not establish
that every optional feature works with every framework version combination.
