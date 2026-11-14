# User guide

## Sources and consumers

A **source** turns a framework function into a gradbridge function; a **consumer** exposes a
gradbridge function to a framework.

| Sources | Consumers |
|---|---|
| `from_torch`, `from_jax`, `from_tf`, `from_warp`, `from_numpy` | `to_torch`, `to_jax`, `to_tf`, `to_warp`, `to_numpy` |

```python
import gradbridge as gb

f = gb.from_tf(tf_fn)            # any source
g = gb.to_jax(f)                 # any consumer
jax.jit(jax.grad(lambda x: g(x).sum()))(x)

g = gb.convert(tf_fn, src="tf", dst="jax")   # the same, in one step
```

A function called from its own framework (PyTorch from PyTorch, JAX from JAX, ...) is composed
natively when the source retains its native callable. Explicit source placement options, such
as `from_torch(..., device=...)`, can select the bridge instead.

## What works

| Consumer | First-order | Compiled | Batched | Second-order | Forward mode |
|---|---|---|---|---|---|
| **PyTorch** | `.backward()`, `torch.autograd.grad` | `torch.compile(fullgraph=True)`, including first-order JVPs / `jacfwd` | `torch.func.vmap`, `vmap(grad(...))` | eager double backward (`create_graph=True`) | `torch.func.jvp`, `jacfwd`, `torch.autograd.forward_ad` |
| **JAX** | `jax.grad`, `jax.vjp`, `jacrev` | `jax.jit` | `jax.vmap` | `grad(grad(...))`, also under `jit` | `to_jax(f, mode="jvp")`: `jax.jvp`, `jacfwd` |
| **TensorFlow** | `GradientTape` | `tf.function` (static callback shapes); XLA for exportable JAX sources with `native=True` | `tf.map_fn` | nested tapes | explicit `to_tf(f.jvp_function)` |
| **NumPy** | `.vjp()`, `.grad()` | — | — | via `vjp_function` | `.jvp()` |
| **Warp** | `wp.Tape` + `tape.backward()` | — | — | — | explicit `to_warp(f.jvp_function)` |

How far derivatives go depends on the **source**:

| Source | Max order | How |
|---|---|---|
| PyTorch | any | `torch.autograd.grad(create_graph=True)` |
| JAX | any | `jax.vjp` of `jax.vjp` |
| TensorFlow | any | nested `GradientTape` |
| NumPy | 1 exact (your `vjp=`), higher by finite differences | or pass a `DiffFunction` as `vjp=` |
| Warp | 1 | Warp's tape is first-order; order 2 raises `gb.MaxOrderError` |

These capabilities also depend on the operations and derivative rules inside the source.
Compiled higher-order PyTorch differentiation is unsupported. Native JAX→TensorFlow graph
lowering is tested with first and second derivatives on CPU/GPU; it requires compatible
serialization versions. See [limitations](limitations.md) for the remaining restrictions.

### Forward-mode sensitivities from PyTorch

`to_torch` uses the source's JVP rule automatically. For example, evaluate the input
sensitivities of a TensorFlow-backed Keras model from PyTorch:

```python
import gradbridge as gb
import gradbridge.torch         # load PyTorch before TensorFlow
import torch
import keras

model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(8, activation="tanh")])
predict = gb.to_torch(gb.from_tf(lambda x: model(x, training=False)))
x = torch.tensor([[0.3, 0.7]])
direction = torch.tensor([[0.2, -0.1]])
value, sensitivity = torch.func.jvp(predict, (x,), (direction,))
jacobian = torch.func.jacfwd(predict)(x)
```

These transforms also compose with `vmap` and with reverse mode when the source can
differentiate its derivative rules. A missing source JVP raises `MaxOrderError` (Warp has
no JVP); NumPy defaults to a directional finite difference unless you provide `jvp=`.
First-order JVPs and `jacfwd` also work when the complete transform is compiled, for example
`torch.compile(torch.func.jacfwd(predict))`. Both `aot_eager` and Inductor are tested. Compile
the transform together with the function, rather than applying an eager JVP to an already
compiled function. Nested compiled JVPs and compiled double backward
remain unsupported; eager higher orders still work. TensorFlow and Warp callers can evaluate
the JVP rule directly with `to_tf(f.jvp_function)` / `to_warp(f.jvp_function)` and arguments
`(*primals, *tangents)`; this does not use automatic `ForwardAccumulator` integration.
Downstream libraries can impose their own restrictions: the Diffrax integration uses
`DirectAdjoint` so the same solve supports both JVPs and reverse gradients.

### Selecting forward mode in JAX

For a foreign source, `to_jax(f)` defaults to a reverse-mode `custom_vjp` wrapper. Select
`mode="jvp"` for `jax.jvp` and `jax.jacfwd`:

```python
import gradbridge as gb
import jax
import jax.numpy as jnp
import torch

source = gb.from_torch(lambda x: torch.sin(x) ** 2)
forward = gb.to_jax(source, mode="jvp")
x = jnp.array([0.2, 0.5], dtype=jnp.float32)
value, tangent = jax.jit(lambda x, v: jax.jvp(forward, (x,), (v,)))(x, jnp.ones_like(x))
```

The forward wrapper does not enable reverse differentiation through an opaque callback;
use a separate default `to_jax(source)` wrapper for `jax.grad` and `jax.vjp`. Both modes
support `jit` and `vmap`, subject to the source rules and batching restrictions.

## Inference and residual caching

In eager PyTorch, calls under `no_grad()` / `inference_mode()`, or without any tensor
requiring gradients, evaluate directly without recording a source graph or tape for backward.
This does not disable forward-mode AD under `no_grad()`. See the
[downstream validation](downstream.md) for reproducible tests and inference timings.

When reverse gradients are needed, eager `to_torch` caches the built-in sources' residuals
by default. `cache_residuals=False` trades that saved state for forward recomputation in
backward. This option applies to the eager bridge; the `torch.compile` and `torch.func`
callback paths use derivative rules that rebuild the source forward state.

## Native TensorFlow graphs

For an exportable JAX function, `native=True` uses `jax2tf` to produce a TensorFlow graph.
Compile the whole TensorFlow computation to let XLA optimize the exported function together
with its caller:

```python
import gradbridge as gb
import jax.numpy as jnp
import tensorflow as tf

source = gb.from_jax(lambda x: jnp.sin(x) ** 2)
physics = gb.to_tf(source, native=True)

@tf.function(jit_compile=True, autograph=False)
def value_and_gradient(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(physics(x) * x + 1)
    return loss, tape.gradient(loss, x)

value, gradient = value_and_gradient(tf.constant([0.2, 0.5], dtype=tf.float32))
```

This path requires compatible JAX/TensorFlow serialization versions and an exportable
function. It has no gradbridge callback at execution, and export errors are reported without
silently switching paths. `native=True` accepts JAX and TensorFlow sources; TensorFlow sources
already compose natively. Other source frameworks use the default callback path, which
supports static-shape `tf.function` calls but has no XLA lowering.

First and second derivatives and XLA execution are verified on CPU/GPU. The
[graph benchmark](limitations-audit.md#native-tensorflow-graph-measurement) includes measured
speedup, optimized-program evidence of fusion and a reproduction command.

## Device handoffs

Supported conversions borrow input buffers through DLPack or native views. A native callback
can still copy a source result into the consumer's output buffer on the same device; avoiding
a host round trip does not mean every operation is copy-free.

| Consumer path | Current behavior |
|---|---|
| JAX calling a PyTorch source | Native CPU/CUDA buffers, including scalar, empty, high-rank and natively batched arrays. The compiled handler does not require Warp. |
| JAX calling another GPU-capable source | Native handlers use Warp's array representation for supported 1–4 dimensional signatures. Non-PyTorch `batched=True` sources and unsupported signatures use host callbacks. |
| TensorFlow callback calling PyTorch | Native CPU/CUDA buffers of arbitrary rank when the compiled handler is available. |
| TensorFlow callback fallback | Supported GPU buffers pass through DLPack without a C++ compiler. JAX/TensorFlow-only configurations do not require Warp for this fallback. |
| NumPy source or consumer | Computation or returned NumPy arrays reside on the host. |

The compiled JAX and TensorFlow handlers currently accept `bool`, `int8`, `uint8`, `int32`,
`int64`, `float16`, `float32` and `float64`; individual source operations can impose further
constraints. TensorFlow's non-PyTorch native handler uses Warp and its rank restrictions.
Contiguity, alignment, dtype or device changes can require copies. Some handoffs also wait
on the host for device work; see [data movement and synchronization](limitations.md#data-movement).

The native handlers compile on first use and are cached under `~/.cache/gradbridge`.
`CXX` selects the compiler and `GRADBRIDGE_CACHE_DIR` selects the cache. See the
[fallback behavior](limitations.md#data-movement) before relying on compiler-free execution.

## A Warp simulation, optimized with JAX

```python
def simulate(v0):                       # launches Warp kernels
    ...
    return final_position

sim = gb.to_jax(gb.from_warp(simulate, in_dtypes=[wp.vec2d]))
grad = jax.jit(jax.grad(lambda v: jnp.sum((sim(v.reshape(1, 2))[0] - target) ** 2)))
```

`in_dtypes` maps `(N, 2)`/`(N, 3)` arrays to Warp vector arrays. Warp functions must allocate
their outputs with `requires_grad=True` so Warp can store their adjoints.

### Many small kernels: `capture=True`

A simulator that launches hundreds of small kernels per call spends most of its time in Python,
once per launch, forward and backward. `gb.from_warp(simulate, capture=True)` records the
forward as a CUDA graph after an eager warmup for a given input signature, and captures the
backward after its initial execution. It then replays those graphs from PyTorch or JAX:

```python
sim = gb.to_torch(gb.from_warp(scene.rollout, in_dtypes=[wp.vec3], capture=True))
```

On Newton's `example_diffsim_ball` (288 substeps), the current RTX 5090 measurement takes a
forward + backward iteration from 69.2 ms to 4.2 ms, compared with 4.0 ms for Newton's native
CUDA graph. All six measured implementations produce bit-identical input gradients
([measurement and reproduction](../benchmarks/RESULTS.md#warp-simulation-and-policy-workloads)).
The example also times a hand-written bridge with hand-captured graphs: all three match. What
`capture=True` saves is writing and maintaining that capture code, and the cases it gets wrong.

The contract: **every call must launch the same kernels, on the same arrays, with the same
scalar arguments.** A replay re-issues the recorded launches; it does not run your Python
again. Arrays your function reads are read at replay time (a simulator's state buffers are
fine), but a scalar that changes between calls (a time `t`, a random seed, a step counter) is
replayed with its captured value, silently. Pass such values as input arrays instead; a GPU
regression checks changing controls across captured values and gradients. Each input signature
(shape, dtype, gradient requirement and device) gets its own graph (up to 8), each holding one
copy of the forward's intermediates. Fixed input buffers are filled on replay, and outputs
are cloned before a later replay can overwrite them.
If capture fails (for example, your function reads an array on the host), gradbridge warns once
and runs that function without graphs. Check a new function once with `gb.gradcheck` or against
`capture=False` before relying on it.

### External writes to Warp intermediates

Later bridged Warp calls automatically invalidate cached tapes whose memory they may have
overwritten. External writes need an explicit notification after the write and before backward:

```python
prediction = simulate(x)
workspace.fill_(0.0)                  # external write to a Warp intermediate
gb.invalidate_warp_residuals(workspace)
prediction.sum().backward()
```

Here `simulate` is a PyTorch consumer of the Warp simulation, and `workspace` is a Warp
intermediate that its forward reconstructs from its saved inputs. Notification inspects
addresses without data copies or GPU synchronization. Tapes with overlapping or unknown
memory footprints are invalidated; independent tapes retain their residuals. Pass several
arrays to notify one group of writes, or no arrays to invalidate every cached Warp tape.

Invalidation does not snapshot hidden model state or restore overwritten inputs. Forwards
must reconstruct the same intermediates from their original inputs; pass changing model
inputs explicitly. If writes cannot be annotated, `to_torch(..., cache_residuals=False)`
always recomputes, with the same reconstruction requirement. Callers still need to order
external writes correctly when using different CUDA streams.

## PyTorch networks inside other frameworks

Tensors a PyTorch function captures, such as an `nn.Module`'s parameters, receive gradients
whichever framework drives the call, accumulated into `.grad` like `loss.backward()`. A PyTorch
network inside a Warp, JAX or TensorFlow computation therefore trains through it:

```python
net = torch.nn.Sequential(...).cuda()
force = gb.to_warp(gb.from_torch(lambda v: net(v)))   # called inside a Warp simulation
...
tape.backward(loss)          # fills net's .grad
optimizer.step()
```

## NumPy functions

NumPy has no autodiff. Supply VJP and JVP rules for exact first derivatives:

```python
import gradbridge as gb
import numpy as np

f = gb.from_numpy(
    lambda x: x ** 3,
    vjp=lambda x, ct: (3 * x ** 2 * ct,),
    jvp=lambda x, tangent: 3 * x ** 2 * tangent,
)
numpy_f = gb.to_numpy(f)
x = np.array([0.2, 0.5])
value = numpy_f(x)
input_cotangents = numpy_f.vjp(x, cotangents=np.ones_like(x))
output_tangent = numpy_f.jvp(x, tangents=np.ones_like(x))
```

Without `vjp=`, the first VJP evaluation warns and uses finite differences. Without `jvp=`,
JVPs use directional finite differences. Plain callable rules do not automatically acquire
exact higher derivatives: pass a differentiable `DiffFunction` as the rule for that support.
NumPy/SciPy computation stays on the CPU even when called from GPU tensors; the consumer
returns results and derivatives to its device.

## Options

- `gb.to_torch(f, cache_residuals=False)` recomputes the forward inside backward instead of
  keeping the source's residuals (its autograd graph, `jax.vjp` pullback or Warp tape).
  This option controls the eager bridge only.
- `gb.invalidate_warp_residuals(*arrays)` notifies gradbridge after external Warp writes, so
  affected tapes recompute. With no arrays, it invalidates all cached Warp tapes. See the
  [reconstruction contract](#external-writes-to-warp-intermediates).
- `gb.to_tf(jax_source, native=True)` exports a native TensorFlow graph through `jax2tf`.
  It supports XLA compilation and tested first/second derivatives for exportable functions and compatible
  JAX/TensorFlow versions; the default callback path remains available for other functions.
- `gb.to_jax(f, mode="jvp")` selects forward mode; the default `mode="vjp"` selects reverse mode.
- `from_*(..., batched=True)` declares that a function accepts a leading batch dimension, so
  the consumer can call the whole batch instead of looping over primal evaluations. This
  does not automatically vectorize a scalar-only function or every derivative rule.
- `from_torch(..., device=...)`, `from_warp(..., device=...)` and `to_warp(..., device=...)`
  select placement where those adapters expose it. CPU↔CUDA placement is verified; peer-device
  placement across two physical GPUs is not yet validated.
- `warnings.simplefilter("error", gb.HostCopyWarning)` turns any unexpected device→host copy
  into an error, useful in tests.

## Check gradients the same way everywhere

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

## Join the protocol from your own library

Any object with a `__gradbridge__()` method returning a `gb.DiffFunction` works with every
consumer, the way `__dlpack__` does for memory:

```python
class MySimulator:
    def __gradbridge__(self):
        return gb.from_warp(self._step)

torch_sim = gb.to_torch(MySimulator())
```
