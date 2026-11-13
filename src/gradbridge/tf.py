"""TensorFlow adapter.

``from_tf`` turns a TensorFlow function into a DiffFunction whose VJP is built
with ``tf.GradientTape`` (nested tapes give higher orders) and whose JVP uses
``tf.autodiff.ForwardAccumulator``.

``to_tf`` exposes any DiffFunction to TensorFlow via ``tf.custom_gradient``
around gradbridge's ``GradbridgeCall`` op, which hands the foreign function
TensorFlow's own (GPU) buffers zero-copy; it is compiled on first use (see
``_tf_native.py``). Without a C++ compiler, or for 0-d / >4-d arrays and
unsupported dtypes, ``tf.py_function`` is used instead, which runs on the host.
A TensorFlow source is composed directly. It works eagerly, inside
``tf.function`` (with static input shapes), under nested ``GradientTape``s and
``tf.map_fn``. It does not work under XLA (``jit_compile=True``).
``tf.vectorized_map`` evaluates it (via TF's loop fallback) but TensorFlow
cannot backpropagate through that fallback -- use ``tf.map_fn`` for gradients.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from typing import Callable, Optional

import numpy as np

# If PyTorch is in use, load torch._dynamo (and Triton's LLVM) before
# TensorFlow's LLVM, otherwise later PyTorch custom ops abort the process.
if "tensorflow" not in sys.modules and "torch" in sys.modules and importlib.util.find_spec("triton") is not None:
    import torch._dynamo  # noqa: F401

import tensorflow as tf

from . import _convert
from . import _tf_native as _tf_op
from ._core import DiffFunction, Spec, _as_tuple, as_diff_function, infer_n_inputs, split_primals

__all__ = ["from_tf", "to_tf"]

_ZERO = tf.UnconnectedGradients.ZERO


def _is_float(t) -> bool:
    return t.dtype.is_floating or t.dtype.is_complex


# =============================================================================
# Source: TensorFlow function -> DiffFunction
# =============================================================================
def _make(fn_t: Callable, *, name, n_inputs=None, n_outputs=None, batched=False):
    def to_t(x):
        return _convert.to_tf(x)

    def _eval(*xs):
        return _as_tuple(fn_t(*[to_t(x) for x in xs]))

    def _fwd(*xs):
        txs = [to_t(x) for x in xs]
        with tf.GradientTape(persistent=True) as tape:
            for x in txs:
                if _is_float(x):
                    tape.watch(x)
            outs = _as_tuple(fn_t(*txs))
        return outs, (tape, txs, outs)

    def _bwd(res, *cts):
        tape, txs, outs = res
        grads = tape.gradient(
            list(outs),
            txs,
            output_gradients=[tf.cast(to_t(c), o.dtype) for c, o in zip(cts, outs)],
            unconnected_gradients=_ZERO,
        )
        return tuple(g if g is not None else tf.zeros_like(x) for g, x in zip(grads, txs))

    def _spec(*specs):
        try:
            conc = tf.function(fn_t).get_concrete_function(
                *[tf.TensorSpec(s.shape, tf.as_dtype(s.dtype)) for s in specs]
            )
            outs = _as_tuple(conc.structured_outputs)
            if all(o.shape.is_fully_defined() for o in outs):
                return tuple(Spec(tuple(o.shape), o.dtype.as_numpy_dtype) for o in outs)
        except Exception:
            pass
        outs = _eval(*[s.zeros() for s in specs])
        return tuple(Spec(tuple(o.shape), o.dtype.as_numpy_dtype) for o in outs)

    dfn = DiffFunction(
        _eval,
        spec=_spec,
        fwd=_fwd,
        bwd=_bwd,
        batched=batched,
        framework="tensorflow",
        name=name,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )

    def vjp_body(*args):
        xs, cts = split_primals(dfn, args)
        txs = [to_t(x) for x in xs]
        with tf.GradientTape() as tape:
            for x in txs:
                if _is_float(x):
                    tape.watch(x)
            outs = _as_tuple(fn_t(*txs))
        grads = tape.gradient(
            list(outs),
            txs,
            output_gradients=[tf.cast(to_t(c), o.dtype) for c, o in zip(cts, outs)],
            unconnected_gradients=_ZERO,
        )
        return tuple(g if g is not None else tf.zeros_like(x) for g, x in zip(grads, txs))

    def jvp_body(*args):
        k = len(args) // 2
        xs = [to_t(x) for x in args[:k]]
        ts = [tf.cast(to_t(t), x.dtype) for x, t in zip(xs, args[k:])]
        with tf.autodiff.ForwardAccumulator(primals=xs, tangents=ts) as acc:
            outs = _as_tuple(fn_t(*xs))
        return tuple(acc.jvp(o, unconnected_gradients=_ZERO) for o in outs)

    dfn._vjp_factory = lambda: _make(vjp_body, name=f"{name}_vjp", n_outputs=dfn.n_inputs)
    dfn._jvp_factory = lambda: _make(jvp_body, name=f"{name}_jvp")
    return dfn


def from_tf(
    fn: Callable,
    *,
    batched: bool = False,
    n_inputs: Optional[int] = None,
    name: Optional[str] = None,
) -> DiffFunction:
    """Wrap a TensorFlow function ``fn(*tensors) -> tensor | tuple``."""
    dfn = _make(
        fn,
        name=name or getattr(fn, "__name__", "tf_fn"),
        n_inputs=infer_n_inputs(fn, n_inputs),
        batched=batched,
    )
    dfn._tf_native = fn
    return dfn


# =============================================================================
# Consumer: DiffFunction -> TensorFlow function
# =============================================================================
def _wrapped(dfn: DiffFunction):
    cache = dfn._consumer_cache
    if "tf_fn" in cache:
        return cache["tf_fn"]

    @tf.custom_gradient
    def f(*xs):
        for x in xs:
            if not x.shape.is_fully_defined():
                raise ValueError(
                    f"gradbridge.to_tf('{dfn.name}'): input shapes must be static inside tf.function "
                    f"(got {x.shape}). Give tf.function a fully-defined input_signature."
                )
        in_specs = [Spec(tuple(x.shape), x.dtype.as_numpy_dtype) for x in xs]
        out_specs = dfn.spec(*in_specs)

        def host(*args):
            outs = dfn.eval(*[a.numpy() for a in args])
            return [
                np.asarray(_convert.to_numpy(o), dtype=s.dtype).reshape(s.shape) for o, s in zip(outs, out_specs)
            ]

        tout = [tf.as_dtype(s.dtype) for s in out_specs]
        specs = [(s.shape, s.dtype) for s in in_specs + list(out_specs)]
        op = _tf_op.load() if _tf_op.supported(specs) else None
        if op is not None:
            # TensorFlow's own buffers and stream, zero-copy (see _tf_native.py).
            try:
                outs = op.gradbridge_call(
                    ins=list(xs),
                    Tout=tout,
                    callback_id=_tf_op.callback_id(dfn),
                    output_shapes=[list(s.shape) for s in out_specs],
                )
            except tf.errors.InternalError:
                _tf_op.raise_pending()
                raise
        else:
            outs = tf.py_function(host, inp=list(xs), Tout=tout)
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        for o, s in zip(outs, out_specs):
            o.set_shape(s.shape)

        def grad(*dys):
            g = _wrapped(dfn.vjp_function)
            in_cts = _as_tuple(g(*xs, *[tf.cast(d, o.dtype) for d, o in zip(dys, outs)]))
            return [tf.cast(c, x.dtype) if _is_float(x) else None for c, x in zip(in_cts, xs)]

        return (outs[0] if len(outs) == 1 else tuple(outs)), grad

    cache["tf_fn"] = f
    return f


def to_tf(fn) -> Callable:
    """Expose any gradbridge function to TensorFlow (``GradientTape`` gradients)."""
    dfn = as_diff_function(fn)
    f = _wrapped(dfn)
    native = dfn._tf_native

    def wrapper(*xs):
        if native is not None:
            # A TensorFlow source called from TensorFlow needs no bridge: compose it directly.
            outs = _as_tuple(native(*[x if isinstance(x, (tf.Tensor, tf.Variable)) else _convert.to_tf(x) for x in xs]))
            return outs[0] if len(outs) == 1 else outs
        outs = f(*[x if isinstance(x, (tf.Tensor, tf.Variable)) else _convert.to_tf(x) for x in xs])
        return outs if not isinstance(outs, (list, tuple)) or len(outs) != 1 else outs[0]

    functools.update_wrapper(wrapper, dfn._eval, assigned=("__doc__",), updated=())
    wrapper.__name__ = dfn.name
    wrapper.__gradbridge__ = lambda: dfn
    return wrapper
