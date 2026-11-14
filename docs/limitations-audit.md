# Limitations audit

Audited on 2026-11-13 against the [tested environment](downstream.md#tested-environment)
and an RTX 5090. This document records current support, measured performance and viable
paths for restrictions that need narrower contracts, compiler integration or more hardware.

Validation: **505 tests passed, zero skipped**, including 71 new regression checks, with
unexpected `HostCopyWarning`s treated as errors. Versions and hardware are recorded in
[downstream validation](downstream.md). New checks live in
[test_limitation_fixes.py](../tests/test_limitation_fixes.py).

## Supported behavior

- Compiling a first-order PyTorch JVP or forward Jacobian works with `aot_eager` and
  Inductor. The compiler path explicitly calls primal and JVP custom ops. Nested compiled
  JVPs raise an error.
- PyTorch functions called from JAX now use native CPU/GPU buffers for scalar, empty and
  high-rank arrays. Natively batched PyTorch functions and their derivative rules run as
  whole batches. These paths also work without Warp. TensorFlow's native PyTorch path
  likewise supports arbitrary tensor rank.
- `gb.to_tf(jax_source, native=True)` exports a native TensorFlow graph with JAX's
  [`jax2tf` interface](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md).
  Exportable functions support TensorFlow XLA and first/second derivatives on CPU/GPU.
  The optimized GPU program fused JAX math with the surrounding TensorFlow operations.
  Arbitrary foreign callbacks are outside this opt-in export contract.
- TensorFlow's fallback no longer requires NumPy conversions. Its Python callback can
  carry GPU buffers through DLPack without a C++ compiler. Tests check identical GPU input
  and output pointers, including after a simulated compiler failure. TensorFlow can label
  callback inputs as CPU even when their actual DLPack allocation is on the GPU; the
  conversions now use the allocation's device. The compiled handler remains the default.
  JAX/TensorFlow-only configurations can use this fallback without requiring Warp.
- `gb.invalidate_warp_residuals(*arrays)` selectively invalidates tapes after external
  buffer writes. It inspects addresses without copying data or synchronizing the GPU.
  Overlapping or unknown memory footprints are invalidated; independent tapes keep their
  residuals. Passing no arrays invalidates every cached tape. External stream ordering is
  still the caller's responsibility.
  Recomputed forwards must reconstruct intermediates from their saved inputs; this is
  cache invalidation rather than a snapshot of arbitrary hidden state.
- Conversion of an existing Warp array honors an explicit destination device.
  CPU↔CUDA transfers and same-device identity are
  verified; this does not substitute for testing two physical GPUs.
- Explicit JVP evaluation in TensorFlow and Warp is supported through the existing
  protocol: `gb.to_tf(dfn.jvp_function)(*xs, *tangents)` or
  `gb.to_warp(dfn.jvp_function)(*xs, *tangents)`. CPU/GPU tests cover both consumers and
  PyTorch, JAX, TensorFlow and NumPy sources. Their automatic forward-mode APIs remain a
  separate integration task.

## Current support and remaining work

| Area | Current support | Remaining work and viable path |
|---|---|---|
| Callback overhead and fusion | Native buffers and batching reduce dispatch and transfer costs. JAX→TensorFlow graph export removes the callback; GPU fusion is verified. | Other source/consumer pairs need native lowering. Coarse functions amortize callback overhead today. |
| Warp kernels inside JAX loops | Whole Warp functions run through a callback and tape. For a single kernel, Warp's [`jax_kernel(..., enable_backward=True)`](https://nvidia.github.io/warp/stable/user_guide/interoperability/jax.html) avoids the Python callback. | A dedicated gradbridge kernel adapter could use direct lowering. Arbitrary Python simulators need the function bridge or explicit graph integration. |
| Changing captured controls | Array-valued controls preserve replay speed; GPU tests check changed values and exact gradients across replays. | Closure scalars stay captured. An API using CUDA [graph parameter updates](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) is viable but not implemented. |
| NumPy placement | `from_numpy` executes ordinary NumPy/SciPy code on the CPU and returns results to the consumer's device. | An opt-in [GPU array/dispatch backend](https://numpy.org/doc/stable/user/basics.interoperability.html) is viable for compatible operations, not arbitrary CPU-only functions. |
| TensorFlow compiler dependency | The default C++ handler is faster; the compiler-free DLPack callback preserves supported GPU allocations. Pointer identity is tested. | Unsupported DLPack signatures can still copy. Prebuilt handlers could remove first-use compilation. |
| Higher derivatives | PyTorch, JAX and TensorFlow support differentiable derivative rules where source operations allow them. Custom `DiffFunction` rules can provide exact higher orders. | Warp's tape remains first-order; general higher-order AD needs [source/compiler work](https://github.com/NVIDIA/warp/issues/1813). Approximate finite-difference HVPs would need an explicit API and error controls. |
| TensorFlow/Warp forward mode | Explicit `dfn.jvp_function` execution is CPU/GPU tested with PyTorch, JAX, TensorFlow and NumPy sources. | Automatic foreign-callback forward-mode integration is unfinished. TensorFlow integration must preserve captured-parameter gradient behavior; Warp sources also need a source JVP. |
| Compiled PyTorch derivatives | Complete first-order JVPs and `jacfwd` transforms compile with `aot_eager` and Inductor. Eager higher derivatives work within source limits. | Nested compiled JVPs are rejected. Compiled double backward is unsupported, including by the tested native [AOTAutograd runtime](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_functorch/_aot_autograd/runtime_wrappers.py). |
| External Warp writes | Selective invalidation is implemented without GPU synchronization. Recomputed forwards must reconstruct intermediates from saved inputs. | Unreported writes remain invisible. Automatic arbitrary-write detection or snapshots need additional instrumentation and cost. |
| Import order | Import PyTorch/`gradbridge.torch` before TensorFlow to satisfy the protective guard. TensorFlow-first CUDA compilation succeeds in an isolated probe with the tested binaries. | A build compatibility matrix or symbol isolation is needed to narrow the guard safely; the issue depends on binary combinations. |
| Multi-GPU | Explicit destination conversion and CPU↔CUDA placement are verified on one physical GPU. | Peer-copy, ownership and stream correctness need two-GPU tests before multi-GPU support can be advertised. |

## Other restrictions in the detailed guide

- **Forward recomputation under compilers:** tensor residuals could be returned as explicit
  outputs and saved by a consumer. Arbitrary Python tapes/graphs cannot be put into XLA or
  TensorFlow tensor values. A managed-handle design also needs cancellation, dead-code,
  repeated-backward and concurrent-call lifetimes. This is viable for selected sources;
  unchecked handle caching would trade correctness or memory safety for speed. The new
  native JAX→TensorFlow graph makes forward/backward math visible to the same optimizer,
  which can reuse common intermediates instead of issuing separate callbacks.
- **Remaining host waits:** device events or running on the consumer's stream can replace
  waits when the producing framework exposes the necessary stream/lifetime interface.
  PyTorch native callbacks already use the consumer's stream. TensorFlow's legacy DLPack
  interface still exports finished buffers; removing that wait needs a different integration,
  not simply deleting the synchronization.
- **Remaining JAX host fallbacks:** the new arbitrary-rank and native batching paths apply to
  PyTorch sources. Other foreign sources still use the Warp representation or host callback
  where that representation cannot express a signature. A general DLPack buffer carrier is
  a viable extension. Unsupported dtypes and source-specific rank limits still need checks.
- **TensorFlow XLA:** the callback op has no `XLA_GPU_JIT` lowering. The new
  `to_tf(..., native=True)` bypasses that op for exportable JAX functions; first/second
  derivatives and CPU/GPU XLA execution are verified. Other sources need their own lowering
  or a compatible XLA custom call. TensorFlow also documents that
  [`py_function` cannot be XLA-compiled](https://www.tensorflow.org/api_docs/python/tf/py_function).
- **Dynamic TensorFlow shapes:** runtime shape inference plus runtime output allocation is
  viable. The current op allocates from static output-shape attributes. A dynamic path needs
  a metadata/allocation protocol; it should retain the existing static fast path.
- **TensorFlow callback `vectorized_map` gradients:** a native batching rule or vectorized whole-batch
  source is preferable to the current loop fallback. That requires TensorFlow's batching and
  gradient machinery, beyond its ordinary custom-gradient registration. `map_fn` remains the
  tested gradient-capable alternative.
- **Outer `torch.cuda.graph` capture:** using a shared stream, preloading kernels, fixed
  buffers and a supported external-capture protocol is a viable specialized route. Current
  arbitrary callbacks and cross-stream ordering are not generally capturable. Warp's internal
  `capture=True` remains the tested integration; this audit does not claim outer capture works.

## Measured effect of removing host fallbacks

RTX 5090, PyTorch 2.10.0+cu128, JAX 0.12.0.1, 1,048,576 float32 elements, a value and gradient
per call. Both paths use the same current derivative rules. The baseline explicitly forces
the host callback to compare placement while keeping the derivative rules unchanged.
Batched derivative vectorization is available to both paths. This is a controlled comparison
within the current implementation, not a comparison of different releases.

| Signature | Forced host fallback (ms) | Native GPU callback (ms) | Speedup |
|---|---:|---:|---:|
| Scalar loss | 2.008 | 0.353 | 5.69× |
| Batched input | 6.672 | 0.340 | 19.61× |
| Five-dimensional input | 14.101 | 0.348 | 40.48× |

Compilation and allocation warmup are excluded. Outputs and gradients are compared before
timing. Each result is the median of seven alternating rounds of ten synchronized calls.
These numbers describe the tested signatures and hardware, not a universal bridge speedup.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
    python benchmarks/bench_jax_native.py --markdown native-callbacks.md
```

## Native TensorFlow graph measurement

On the same GPU and versions, a 1,048,576-element float32 computation evaluates
`sum(sin(x)**2 * x + 1)` and its gradient. The default device callback runs inside
`tf.function`; `native=True` allows compiling the entire TensorFlow computation with XLA.
The comparison includes that newly available compiler optimization.

| Device callback (ms) | Native graph + XLA (ms) | Speedup |
|---:|---:|---:|
| 0.823 | 0.145 | 5.68× |

Values and full gradients were compared before timing. Warmup is excluded; measurements
use seven alternating rounds of ten synchronized calls. The optimized HLO's
`fused_add_reduce` computation contains both JAX's sine/power operations and TensorFlow's
multiply, add, reduction and gradient operations. Fusion is verified for this program,
not promised for every exported function.

```bash
python benchmarks/bench_tf_native.py --markdown native-tf.md --hlo optimized.hlo
```

## Using the APIs

Export a pure JAX function to a native TensorFlow graph, then compile the whole computation:

```python
tf_function = gb.to_tf(gb.from_jax(jax_function), native=True)
compiled_tf = tf.function(tf_function, jit_compile=True, autograph=False)
```

This opt-in path requires compatible JAX/TensorFlow serialization versions and an exportable
function. It does not silently fall back if export fails. The default callback path continues
to support functions that call other foreign frameworks.

Compile the JVP or Jacobian operation together with the function that calls the bridge:

```python
f = gb.to_torch(dfn)
jvp = torch.compile(lambda x, v: torch.func.jvp(f, (x,), (v,)))
jacobian = torch.compile(torch.func.jacfwd(f))
```

Notify after external writes and before backward:

```python
out = simulate(x)
intermediate_buffer.fill_(0.0)             # your own Warp write
gb.invalidate_warp_residuals(intermediate_buffer)
out.sum().backward()                       # recomputes affected forward state
```

The simulation's forward must reconstruct `intermediate_buffer` from its saved inputs.
Notification does not restore hidden inputs or order writes from an external CUDA stream;
see the [full mutation contract](guide.md#external-writes-to-warp-intermediates).

For consumers without automatic forward-mode integration, call the derivative rule directly:

```python
tf_jvp = gb.to_tf(dfn.jvp_function)
tangent_output = tf_jvp(x, direction)       # one input; use *xs, *tangents for several
```

The remaining compiler and hardware work has viable paths. Each needs its own correctness
and performance evidence before it can be removed from the supported-behavior contract.
