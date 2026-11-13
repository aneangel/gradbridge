"""Finite-difference derivative rules for any DiffFunction.

Used as the fallback for sources without autodiff (plain NumPy functions) and
as the reference in :func:`gradbridge.gradcheck`. Central differences, so the
cost is ``2 * (number of input elements)`` evaluations -- fine for checking and
small functions, too slow for large models.
"""

from __future__ import annotations

import numpy as np

from . import _convert
from ._core import DiffFunction, split_primals


def default_eps(dtype) -> float:
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return 1e-6
    if dtype == np.float32:
        return 1e-3
    return 1e-2


def _eval_np(fn: DiffFunction, xs):
    # copy=True: the source may return outputs that alias the inputs we perturb in place.
    return [np.array(_convert.to_numpy(o), dtype=np.float64, copy=True) for o in fn.eval(*xs)]


def fd_vjp(fn: DiffFunction, *xs, cts, eps=None):
    """Central-difference VJP of ``fn`` at ``xs`` against cotangents ``cts``."""
    xs = [np.array(_convert.to_numpy(x)) for x in xs]
    cts = [np.asarray(_convert.to_numpy(c), dtype=np.float64) for c in cts]
    grads = []
    for i, x in enumerate(xs):
        if not np.issubdtype(x.dtype, np.floating):
            grads.append(np.zeros(x.shape, dtype=x.dtype))
            continue
        e = default_eps(x.dtype) if eps is None else eps
        g = np.zeros(x.shape, dtype=np.float64)
        flat = x.reshape(-1)
        for k in range(flat.size):
            orig = flat[k]
            h = e * max(1.0, abs(float(orig)))
            flat[k] = orig + h
            plus = _eval_np(fn, xs)
            flat[k] = orig - h
            minus = _eval_np(fn, xs)
            flat[k] = orig
            g.reshape(-1)[k] = sum(float(np.sum(c * (p - m))) for c, p, m in zip(cts, plus, minus)) / (2 * h)
        grads.append(g.astype(x.dtype))
    return tuple(grads)


def fd_jvp(fn: DiffFunction, *xs, tangents, eps=None):
    """Central-difference JVP: directional derivative of ``fn`` along ``tangents``."""
    xs = [np.asarray(_convert.to_numpy(x)) for x in xs]
    ts = [np.asarray(_convert.to_numpy(t), dtype=np.float64) for t in tangents]
    dtype = next((x.dtype for x in xs if np.issubdtype(x.dtype, np.floating)), np.dtype("float64"))
    e = default_eps(dtype) if eps is None else eps
    plus = _eval_np(fn, [x + e * t if np.issubdtype(x.dtype, np.floating) else x for x, t in zip(xs, ts)])
    minus = _eval_np(fn, [x - e * t if np.issubdtype(x.dtype, np.floating) else x for x, t in zip(xs, ts)])
    return tuple(((p - m) / (2 * e)).astype(dtype) for p, m in zip(plus, minus))


def fd_vjp_function(fn: DiffFunction, eps=None) -> DiffFunction:
    """A DiffFunction ``g(*xs, *cts)`` computing ``fn``'s VJP by finite differences.

    Its own VJP is again finite differences, so arbitrary order works (with
    rapidly growing cost and noise).
    """

    def g_eval(*args):
        xs, cts = split_primals(fn, args)
        return fd_vjp(fn, *xs, cts=cts, eps=eps)

    g = DiffFunction(
        g_eval,
        framework=f"finite-differences({fn.framework})",
        name=f"{fn.name}_fdvjp",
        n_outputs=fn.n_inputs,
    )
    g._vjp_factory = lambda: fd_vjp_function(g, eps=eps)
    return g


def fd_jvp_function(fn: DiffFunction, eps=None) -> DiffFunction:
    def h_eval(*args):
        k = len(args) // 2
        return fd_jvp(fn, *args[:k], tangents=args[k:], eps=eps)

    h = DiffFunction(h_eval, framework=f"finite-differences({fn.framework})", name=f"{fn.name}_fdjvp")
    h._vjp_factory = lambda: fd_vjp_function(h, eps=eps)
    return h
