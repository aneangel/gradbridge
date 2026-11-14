# Known limitations

## Performance

- **Per-call overhead.** A few tenths of a millisecond for small arrays (see the
  [benchmarks](../benchmarks/RESULTS.md)), and nothing fuses across the boundary. Bridge coarse
  functions — a simulator rollout, a model, a loss — not single elementwise operations.
- **Recomputation under compilers.** In eager PyTorch the forward's residuals are kept for the
  backward pass. Under `torch.compile` and `torch.func`, and in the JAX and TensorFlow
  consumers, the backward pass re-runs the forward to rebuild them.
- **Some zero-copy handoffs wait on the host** instead of copying: TensorFlow only exports
  finished tensors and imports without a stream handshake, and the JAX and TensorFlow consumer
  callbacks wait for their framework's stream before returning.
- **Warp kernels called from JAX in a hot loop.** gradbridge calls your Warp *function* (with
  its tape) from an XLA callback, about 0.5–0.9 ms per jitted gradient step. Warp's own
  `wp.jax_kernel` launches a single kernel from inside XLA in about 0.09 ms. When the work is one
  Warp kernel whose outputs have the shape of its launch grid, use `jax_kernel` for the JAX side;
  gradbridge is the better fit for whole functions, other source frameworks, and PyTorch callers.

## Data movement

Arrays cross between PyTorch, JAX, TensorFlow and Warp zero-copy, with each pair's CUDA streams
ordered. These cases copy device→host→device instead:

- NumPy functions (by nature).
- Under the JAX consumer: natively `batched=True` sources, 0-d or >4-d arrays, and dtypes Warp
  has no scalar type for.
- Calls from TensorFlow when its op could not be compiled (see below).

An unexpected fallback emits `gb.HostCopyWarning`. Turn it into an error with
`warnings.simplefilter("error", gb.HostCopyWarning)` to catch regressions.

## TensorFlow

- **Calling other frameworks from TensorFlow needs a C++ compiler for zero-copy.**
  `tf.py_function` is a host op — TensorFlow places its inputs on the CPU even under
  `tf.device("/GPU:0")` — so gradbridge ships a small TensorFlow op (`_tf_op.cc`) that hands the
  bridged function TensorFlow's own GPU buffers. It is compiled on first use against your
  TensorFlow and cached in `~/.cache/gradbridge`; the `CXX` and `GRADBRIDGE_CACHE_DIR`
  environment variables override the compiler and the location. Without a compiler, gradbridge
  warns once and falls back to `tf.py_function`.
- No XLA (`jit_compile=True`), and input shapes must be static inside `tf.function`.
- `tf.vectorized_map` gives correct values, but TensorFlow cannot backpropagate through its loop
  fallback (reproducible without gradbridge). Use `tf.map_fn` when you need gradients.
- **Import order:** TensorFlow and Triton (used by `torch.compile`) bundle conflicting LLVM
  builds. Import `gradbridge.torch` (or PyTorch) before TensorFlow; gradbridge raises a clear
  `ImportError` rather than letting the process abort.

## Derivatives

- **Second order** needs a source that supports it: PyTorch, JAX and TensorFlow go to any order;
  Warp's tape is first-order (order 2 raises `gb.MaxOrderError`); NumPy uses finite differences
  beyond the VJP you provide.
- **Forward mode** is available for the JAX and NumPy consumers only.
- **`torch.compile` together with double backward** is not supported (the compiled backward
  calls an opaque VJP op). Each works on its own.

## Warp buffers

In eager PyTorch, a Warp source's tape is kept for the backward pass. gradbridge records the
memory each Warp forward touched and recomputes any tape that a later bridged Warp call may
have overwritten (for example several rollouts of one simulator before one `backward()`), so
those gradients stay exact. It cannot see *your own* code writing into those buffers between
forward and backward: pass `cache_residuals=False` to `to_torch` to always recompute in that
case.

## Platforms

Tested on Linux, on CPU and on one NVIDIA GPU (RTX 5090). Multi-GPU placement has not been
validated yet.
