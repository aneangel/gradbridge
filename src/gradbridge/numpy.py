"""NumPy adapter: plain NumPy functions in, NumPy-callable gradients out.

NumPy has no autodiff, so a NumPy source either supplies its own VJP/JVP or
falls back to finite differences.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np

from . import _convert, _fd
from ._core import DiffFunction, as_diff_function, infer_n_inputs, split_primals

__all__ = ["from_numpy", "to_numpy"]


def from_numpy(
    fn: Callable,
    *,
    vjp: Optional[Union[Callable, DiffFunction]] = None,
    jvp: Optional[Union[Callable, DiffFunction]] = None,
    batched: bool = False,
    n_inputs: Optional[int] = None,
    name: Optional[str] = None,
    fd_eps: Optional[float] = None,
) -> DiffFunction:
    """Wrap a NumPy function ``fn(*arrays) -> array | tuple``.

    Parameters
    ----------
    vjp:
        ``vjp(*xs, *cts) -> tuple`` of input cotangents (one cotangent per
        output). May itself be a DiffFunction to enable exact higher orders.
        If omitted, finite differences are used (with a one-time warning).
    jvp:
        ``jvp(*xs, *tangents) -> tuple`` of output tangents. Defaults to
        finite differences.
    """
    name = name or getattr(fn, "__name__", "numpy_fn")

    def _eval(*xs):
        return fn(*[_convert.to_numpy(x) for x in xs])

    dfn = DiffFunction(
        _eval,
        batched=batched,
        framework="numpy",
        name=name,
        n_inputs=infer_n_inputs(fn, n_inputs),
    )

    if vjp is None:
        warned = []

        def _fd_factory():
            g = _fd.fd_vjp_function(dfn, eps=fd_eps)
            inner = g._eval

            def eval_with_warning(*args):
                # Warn when finite differences are actually used, not when the
                # rule is merely created (consumers pre-build rules eagerly).
                if not warned:
                    warnings.warn(
                        f"gradbridge: '{name}' has no VJP; using finite differences "
                        "(slow, approximate). Pass vjp= to from_numpy for exact gradients.",
                        stacklevel=2,
                    )
                    warned.append(True)
                return inner(*args)

            g._eval = eval_with_warning
            return g

        dfn._vjp_factory = _fd_factory
    elif isinstance(vjp, DiffFunction) or hasattr(vjp, "__gradbridge__"):
        user = as_diff_function(vjp)
        if user.n_outputs is None:
            user.n_outputs = dfn.n_inputs
        dfn._vjp_factory = lambda: user
    else:
        def _vjp_factory():
            g = from_numpy(vjp, name=f"{name}_vjp", fd_eps=fd_eps)
            g.n_outputs = dfn.n_inputs
            return g

        dfn._vjp_factory = _vjp_factory

    if jvp is None:
        dfn._jvp_factory = lambda: _fd.fd_jvp_function(dfn, eps=fd_eps)
    elif isinstance(jvp, DiffFunction) or hasattr(jvp, "__gradbridge__"):
        dfn._jvp_factory = lambda: as_diff_function(jvp)
    else:
        dfn._jvp_factory = lambda: from_numpy(jvp, name=f"{name}_jvp", fd_eps=fd_eps)
    return dfn


class NumpyFunction:
    """A DiffFunction exposed to NumPy: call it, and ask for VJPs/JVPs explicitly."""

    def __init__(self, dfn: DiffFunction):
        self.diff_function = dfn

    def __call__(self, *xs):
        outs = tuple(np.asarray(_convert.to_numpy(o)) for o in self.diff_function.eval(*xs))
        return outs[0] if len(outs) == 1 else outs

    def vjp(self, *xs, cotangents):
        """Input cotangents for the given output cotangents (tuple, one per input)."""
        cts = cotangents if isinstance(cotangents, (tuple, list)) else (cotangents,)
        res = self.diff_function.vjp_function.eval(*xs, *cts)
        return tuple(np.asarray(_convert.to_numpy(r)) for r in res)

    def jvp(self, *xs, tangents):
        """Output tangents for the given input tangents."""
        ts = tangents if isinstance(tangents, (tuple, list)) else (tangents,)
        res = self.diff_function.jvp_function.eval(*xs, *ts)
        out = tuple(np.asarray(_convert.to_numpy(r)) for r in res)
        return out[0] if len(out) == 1 else out

    def grad(self, *xs):
        """Gradient of a scalar-output function w.r.t. every input."""
        outs = self.diff_function.eval(*xs)
        if len(outs) != 1 or np.size(_convert.to_numpy(outs[0])) != 1:
            raise ValueError("grad() needs a function with a single scalar output; use vjp().")
        ct = np.ones_like(np.asarray(_convert.to_numpy(outs[0])))
        g = self.vjp(*xs, cotangents=(ct,))
        return g[0] if len(g) == 1 else g

    def __gradbridge__(self):
        return self.diff_function


def to_numpy(fn) -> NumpyFunction:
    """Expose any gradbridge function to NumPy code."""
    return NumpyFunction(as_diff_function(fn))


# keep split_primals importable from here for user-written NumPy VJPs
split_primals = split_primals
