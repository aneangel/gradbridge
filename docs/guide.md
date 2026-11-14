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
natively: no bridge, native speed.

## What works

| Consumer | First-order | Compiled | Batched | Second-order | Forward mode |
|---|---|---|---|---|---|
| **PyTorch** | `.backward()`, `torch.autograd.grad` | `torch.compile(fullgraph=True)` | `torch.func.vmap`, `vmap(grad(...))` | double backward (`create_graph=True`) | — |
| **JAX** | `jax.grad`, `jax.vjp`, `jacrev` | `jax.jit` | `jax.vmap` | `grad(grad(...))`, also under `jit` | `to_jax(f, mode="jvp")`: `jax.jvp`, `jacfwd` |
| **TensorFlow** | `GradientTape` | `tf.function` (static shapes) | `tf.map_fn` | nested tapes | — |
| **NumPy** | `.vjp()`, `.grad()` | — | — | via `vjp_function` | `.jvp()` |
| **Warp** | `wp.Tape` + `tape.backward()` | — | — | — | — |

How far derivatives go depends on the **source**:

| Source | Max order | How |
|---|---|---|
| PyTorch | any | `torch.autograd.grad(create_graph=True)` |
| JAX | any | `jax.vjp` of `jax.vjp` |
| TensorFlow | any | nested `GradientTape` |
| NumPy | 1 exact (your `vjp=`), higher by finite differences | or pass a `DiffFunction` as `vjp=` |
| Warp | 1 | Warp's tape is first-order; order 2 raises `gb.MaxOrderError` |

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
forward and the backward as CUDA graphs on the second call with a given input shape and replays
them afterwards, from PyTorch or JAX:

```python
sim = gb.to_torch(gb.from_warp(scene.rollout, in_dtypes=[wp.vec3], capture=True))
```

On Newton's `example_diffsim_ball` (288 substeps) this takes a training iteration from 63 ms to
5 ms, the same speed as Newton's own loop with its own CUDA graph, and bit-identical results
([example](https://github.com/aneangel/gradbridge/blob/main/examples/newton_ball_torch.py)).
The example also times a hand-written bridge with hand-captured graphs: all three match. What
`capture=True` saves is writing and maintaining that capture code, and the cases it gets wrong.

The contract: **every call must launch the same kernels, on the same arrays, with the same
scalar arguments.** A replay re-issues the recorded launches; it does not run your Python
again. Arrays your function reads are read at replay time (a simulator's state buffers are
fine), but a scalar that changes between calls (a time `t`, a random seed, a step counter) is
replayed with its captured value, silently. Pass such values as input arrays instead. Each
input shape gets its own graph (up to 8), each holding one copy of the forward's intermediates.
If capture fails (for example, your function reads an array on the host), gradbridge warns once
and runs that function without graphs. Check a new function once with `gb.gradcheck` or against
`capture=False` before relying on it.

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

NumPy has no autodiff, so give a VJP, or gradbridge falls back to finite differences with a
warning:

```python
f = gb.from_numpy(lambda x: x ** 3, vjp=lambda x, ct: (3 * x ** 2 * ct,))
```

## Options

- `gb.to_torch(f, cache_residuals=False)` recomputes the forward inside backward instead of
  keeping the source's residuals (its autograd graph, `jax.vjp` pullback or Warp tape).
- `from_*(..., batched=True)` declares that a function accepts a leading batch dimension, so
  `vmap` makes one call instead of looping.
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
