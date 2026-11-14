"""JAX adapter.

``from_jax`` turns a JAX function into a DiffFunction whose VJP/JVP rules are
``jax.vjp`` / ``jax.jvp`` of the function -- themselves JAX functions, so every
derivative order is available.

``to_jax`` exposes any DiffFunction to JAX via ``jax.custom_vjp`` around a
callback into the foreign framework (see below). It works under ``jax.jit``, ``jax.vmap`` and nested
``jax.grad`` (reverse-over-reverse). With ``mode="jvp"`` it uses
``jax.custom_jvp`` instead, enabling forward mode (``jax.jvp``, ``jax.jacfwd``)
but not reverse mode -- JAX cannot transpose an opaque callback.

The compiled XLA FFI handler passes CPU/CUDA buffers directly to PyTorch sources,
including scalar, high-rank and batched arrays. Other sources use Warp arrays through
that handler or Warp's ``jax_callable``. A JAX source is composed directly. Unsupported
signatures use ``jax.pure_callback`` (or ``io_callback`` for side-effecting VJPs), which
copies device -> host -> device.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp

from . import _convert
from ._core import DiffFunction, Spec, _as_tuple, as_diff_function, infer_n_inputs, split_primals

__all__ = ["from_jax", "to_jax"]


# =============================================================================
# Source: JAX function -> DiffFunction
# =============================================================================
def _make(fn_t: Callable, *, name, n_inputs=None, n_outputs=None, batched=False, jit=True):
    jfn = jax.jit(fn_t) if jit else fn_t

    def _eval(*xs):
        return _as_tuple(jfn(*[_convert.to_jax(x) for x in xs]))

    # jax.vjp's pullback is a pytree (residual arrays + structure), so both the
    # forward-with-residuals and the backward can be jit-compiled.
    def _fwd_impl(*xs):
        return jax.vjp(fn_t, *xs)

    def _bwd_impl(vjp, cts):
        return vjp(cts)

    fwd_c = jax.jit(_fwd_impl) if jit else _fwd_impl
    bwd_c = jax.jit(_bwd_impl) if jit else _bwd_impl

    def _fwd(*xs):
        outs, vjp = fwd_c(*[_convert.to_jax(x) for x in xs])
        return _as_tuple(outs), vjp

    def _bwd(vjp, *cts):
        return tuple(bwd_c(vjp, tuple(_convert.to_jax(c) for c in cts)))

    def _spec(*specs):
        outs = jax.eval_shape(fn_t, *[jax.ShapeDtypeStruct(s.shape, s.dtype) for s in specs])
        return tuple(Spec(o.shape, np.dtype(o.dtype)) for o in _as_tuple(outs))

    dfn = DiffFunction(
        _eval,
        spec=_spec,
        fwd=_fwd,
        bwd=_bwd,
        batched=batched,
        framework="jax",
        name=name,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )

    def vjp_body(*args):
        xs, cts = split_primals(dfn, args)
        _, vjp = jax.vjp(fn_t, *xs)
        res = vjp(tuple(cts))
        # integer inputs get float0 cotangents in JAX; replace with zeros of the input dtype
        return tuple(
            jnp.zeros(jnp.shape(x), dtype=jnp.result_type(x)) if r.dtype == jax.dtypes.float0 else r
            for r, x in zip(res, xs)
        )

    def jvp_body(*args):
        k = len(args) // 2
        xs, ts = tuple(args[:k]), tuple(args[k:])
        _, t_out = jax.jvp(fn_t, xs, ts)
        return _as_tuple(t_out)

    dfn._jax_native = fn_t
    dfn._vjp_factory = lambda: _make(vjp_body, name=f"{name}_vjp", n_outputs=dfn.n_inputs, jit=jit)
    dfn._jvp_factory = lambda: _make(jvp_body, name=f"{name}_jvp", jit=jit)
    return dfn


def from_jax(
    fn: Callable,
    *,
    batched: bool = False,
    n_inputs: Optional[int] = None,
    name: Optional[str] = None,
    jit: bool = True,
) -> DiffFunction:
    """Wrap a JAX function ``fn(*arrays) -> array | tuple``.

    The function is ``jax.jit``-compiled by default. Gradients of any order
    (and forward-mode JVPs) come from JAX itself.
    """

    def fn_t(*xs):
        return _as_tuple(fn(*xs))

    return _make(
        fn_t,
        name=name or getattr(fn, "__name__", "jax_fn"),
        n_inputs=infer_n_inputs(fn, n_inputs),
        batched=batched,
        jit=jit,
    )


# =============================================================================
# Consumer: DiffFunction -> JAX function
# =============================================================================
def _is_float(dtype) -> bool:
    return jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.complexfloating)


def _jax_dtype(np_dtype):
    # Respect jax_enable_x64: float64 specs become float32 when x64 is off.
    return jax.dtypes.canonicalize_dtype(np_dtype)


def _callback(dfn: DiffFunction, xs):
    if dfn._jax_native is not None:
        # A JAX source called from JAX needs no bridge: compose it as ordinary JAX code.
        return _as_tuple(dfn._jax_native(*xs))

    in_specs = [Spec(tuple(x.shape), np.dtype(x.dtype)) for x in xs]
    out_specs = dfn.spec(*in_specs)
    result_shapes = tuple(jax.ShapeDtypeStruct(s.shape, _jax_dtype(s.dtype)) for s in out_specs)

    def host(*args):
        outs = dfn.eval(*args)
        # no reshape: under vmap(broadcast_all) the batch dimension is included
        return tuple(np.asarray(_convert.to_numpy(o), dtype=r.dtype) for o, r in zip(outs, result_shapes))

    vmap_method = "broadcast_all" if dfn.batched else "sequential"

    def host_call(*args):
        if dfn.has_side_effects:
            # io_callback: XLA may neither drop it (when its outputs are unused) nor repeat it.
            from jax.experimental import io_callback

            return io_callback(host, result_shapes, *args, ordered=False)
        return jax.pure_callback(host, result_shapes, *args, vmap_method=vmap_method)

    # The FFI handler runs on CPU and CUDA alike, so no platform switch is needed (a lax switch
    # would also make side-effecting VJPs fail under vmap: "IO effect not supported in vmap-of-cond").
    native_call = _native_callable(dfn, xs, result_shapes)
    if native_call is not None:
        return native_call(*xs)
    device_call = _gpu_callable(dfn, xs, result_shapes)
    if device_call is None:
        return host_call(*xs)
    return device_call(*xs)


def _native_callable(dfn: DiffFunction, xs, result_shapes):
    """gradbridge's own compiled XLA FFI handler (``_jax_op.cc``), or None to fall back."""
    if dfn.framework != "torch" and (dfn.batched or _warp_or_none() is None):
        return None
    from . import _jax_native

    specs = [(tuple(x.shape), x.dtype) for x in xs] + [(tuple(r.shape), r.dtype) for r in result_shapes]
    if not _jax_native.supported(specs, framework=dfn.framework):
        return None
    target = _jax_native.load()
    if target is None:
        return None
    call = jax.ffi.ffi_call(
        target, result_shapes, vmap_method="broadcast_all" if dfn.batched else "sequential",
        has_side_effect=dfn.has_side_effects,
    )
    cid = np.int64(_jax_native.callback_id(dfn))

    def native_call(*args):
        return tuple(_as_tuple(call(*args, callback_id=cid)))

    return native_call


# =============================================================================
# GPU-resident callbacks (XLA FFI through Warp's jax_callable)
# =============================================================================
# pure_callback always hands the callback host arrays. The call instead goes through an XLA FFI
# handler (Warp's public jax_callable), registered for CPU and CUDA: the foreign function receives
# XLA's buffers zero-copy and, on GPU, runs ordered on XLA's stream, so data never leaves the device.
def _warp_or_none():
    try:
        import warp as wp
    except ImportError:  # pragma: no cover - warp is optional
        return None
    return wp if hasattr(wp, "jax_callable") else None


def _ffi_compatible(shapes_dtypes) -> bool:
    """Warp arrays have 1-4 dimensions and scalar Warp dtypes; anything else stays on the host path."""
    for shape, dtype in shapes_dtypes:
        if not 1 <= len(shape) <= 4:
            return False
        try:
            _convert.warp_scalar_dtype(np.dtype(dtype))
        except KeyError:
            return False
    return True


def _gpu_callable(dfn: DiffFunction, xs, result_shapes):
    """The FFI-based callable for this signature, or None if this call must use pure_callback."""
    wp = _warp_or_none()
    if wp is None or dfn.batched:
        # batched sources rely on broadcast_all shapes, which Warp's FFI collapses; keep the host path
        return None
    ins = [(tuple(x.shape), x.dtype) for x in xs]
    outs = [(tuple(r.shape), r.dtype) for r in result_shapes]
    if not _ffi_compatible(ins + outs):
        return None

    key = (tuple(ins), tuple(outs), dfn.has_side_effects)
    cache = dfn._consumer_cache.setdefault("jax_ffi", {})
    if key not in cache:
        cache[key] = _make_ffi_callable(wp, dfn, ins, outs)
    ffi_fn, output_dims = cache[key]

    def device_call(*args):
        return tuple(_as_tuple(ffi_fn(*args, output_dims=output_dims)))

    return device_call


def _make_ffi_callable(wp, dfn: DiffFunction, ins, outs):
    def body(in_arrays, out_arrays):
        # Runs inside XLA's execution with Warp's current stream set to XLA's stream. DLPack
        # conversions in the sources order their streams against it in both directions.
        results = dfn.eval(*in_arrays)
        for dst, res in zip(out_arrays, results):
            src = _convert.to_warp(res, device=dst.device)
            if src.dtype != dst.dtype:
                src = wp.array(np.asarray(_convert.to_numpy(res), dtype=np.dtype(dst.dtype._type_)), device=dst.device)
            wp.copy(dst, src.reshape(dst.shape) if src.shape != dst.shape else src)
        # The results may be freed (and their memory reused by another framework's allocator on
        # its own stream) as soon as this returns; wait until XLA's stream has finished copying.
        # (The handler also runs for CPU computations, e.g. arrays committed to a CPU device.)
        if out_arrays[0].device.is_cuda:
            wp.synchronize_stream(wp.get_stream(out_arrays[0].device))

    n_in, n_out = len(ins), len(outs)
    in_names = [f"x{i}" for i in range(n_in)]
    out_names = [f"y{i}" for i in range(n_out)]
    params = ", ".join(in_names + out_names)
    namespace = {"_body": body}
    exec(  # noqa: S102 - builds a fixed-arity signature; jax_callable reads the annotations
        f"def _bridge({params}):\n    _body(({', '.join(in_names)},), ({', '.join(out_names)},))\n", namespace
    )
    fn = namespace["_bridge"]
    fn.__annotations__ = {
        name: wp.array(dtype=_convert.warp_scalar_dtype(np.dtype(dtype)), ndim=len(shape))
        for name, (shape, dtype) in zip(in_names + out_names, ins + outs)
    }
    fn.__qualname__ = fn.__name__ = f"gradbridge_{dfn.name}"
    ffi_fn = wp.jax_callable(
        fn,
        num_outputs=n_out,
        graph_mode=wp.JaxCallableGraphMode.NONE,
        vmap_method="sequential",
        has_side_effect=dfn.has_side_effects,  # never dropped as dead code, never repeated
    )
    output_dims = {name: shape for name, (shape, _) in zip(out_names, outs)}
    return ffi_fn, output_dims


def _vjp_wrapped(dfn: DiffFunction):
    cache = dfn._consumer_cache
    if "jax_vjp" in cache:
        return cache["jax_vjp"]

    @jax.custom_vjp
    def f(*xs):
        return _callback(dfn, xs)

    def f_fwd(*xs):
        # Call f itself (not the raw callback) so an *outer* differentiation
        # level -- e.g. grad(grad(...)) -- sees a custom_vjp, not a bare
        # pure_callback, which JAX cannot differentiate.
        return f(*xs), xs

    def f_bwd(xs, cts):
        g = _vjp_wrapped(dfn.vjp_function)
        in_cts = g(*xs, *cts)
        out = []
        for ct, x in zip(in_cts, xs):
            if _is_float(x.dtype):
                out.append(ct.astype(x.dtype))
            else:
                out.append(np.zeros(x.shape, dtype=jax.dtypes.float0))
        return tuple(out)

    f.defvjp(f_fwd, f_bwd)
    cache["jax_vjp"] = f
    return f


def _jvp_wrapped(dfn: DiffFunction):
    cache = dfn._consumer_cache
    if "jax_jvp" in cache:
        return cache["jax_jvp"]

    @jax.custom_jvp
    def f(*xs):
        return _callback(dfn, xs)

    @f.defjvp
    def f_jvp(primals, tangents):
        outs = _callback(dfn, primals)
        tangents = tuple(
            jnp.zeros(jnp.shape(p), jnp.result_type(p)) if type(t).__name__ == "Zero" else t
            for p, t in zip(primals, tangents)
        )
        t_out = _jvp_wrapped(dfn.jvp_function)(*primals, *tangents)
        return outs, tuple(t.astype(o.dtype) for t, o in zip(t_out, outs))

    cache["jax_jvp"] = f
    return f


def to_jax(fn, *, mode: str = "vjp") -> Callable:
    """Expose any gradbridge function to JAX.

    ``mode="vjp"`` (default): reverse mode -- ``jax.grad``, ``jax.vjp``,
    ``jax.jacrev``, nested to any order the source supports.
    ``mode="jvp"``: forward mode -- ``jax.jvp``, ``jax.jacfwd``.
    Both work under ``jax.jit`` and ``jax.vmap``.
    """
    dfn = as_diff_function(fn)
    if mode == "vjp":
        f = _vjp_wrapped(dfn)
    elif mode == "jvp":
        f = _jvp_wrapped(dfn)
    else:
        raise ValueError("mode must be 'vjp' or 'jvp'")

    def wrapper(*xs):
        outs = f(*[x if isinstance(x, jax.Array) else jnp.asarray(_convert.to_numpy(x)) for x in xs])
        outs = tuple(outs)
        return outs[0] if len(outs) == 1 else outs

    functools.update_wrapper(wrapper, dfn._eval, assigned=("__doc__",), updated=())
    wrapper.__name__ = dfn.name
    wrapper.__gradbridge__ = lambda: dfn
    return wrapper
