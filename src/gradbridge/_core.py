"""The gradbridge protocol: a framework-neutral differentiable function.

A :class:`DiffFunction` maps a tuple of arrays to a tuple of arrays and knows how
to produce its own derivative rules *as further DiffFunctions*:

* ``vjp_function`` -- ``g(*xs, *cotangents) -> input cotangents`` (reverse mode)
* ``jvp_function`` -- ``h(*xs, *tangents)   -> output tangents``  (forward mode)

Because derivative rules are themselves DiffFunctions, higher-order derivatives
come for free whenever the source framework can differentiate its own rules
(PyTorch, JAX, TensorFlow), and stop with a clear error when it cannot (Warp).

Any object can join the protocol by implementing ``__gradbridge__()`` and
returning a DiffFunction -- the "DLPack for derivatives" idea.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = [
    "Spec",
    "DiffFunction",
    "as_diff_function",
    "split_primals",
    "infer_n_inputs",
    "MaxOrderError",
]

_name_counter = itertools.count()


class MaxOrderError(NotImplementedError):
    """Raised when a derivative order beyond what the source framework supports is requested."""


@dataclass(frozen=True)
class Spec:
    """Shape and dtype of one array crossing the bridge."""

    shape: tuple
    dtype: np.dtype

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def is_float(self) -> bool:
        return np.issubdtype(self.dtype, np.floating) or np.issubdtype(self.dtype, np.complexfloating)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)


def _as_tuple(out) -> tuple:
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


class DiffFunction:
    """A differentiable function that any framework can call and differentiate.

    Parameters
    ----------
    eval:
        ``eval(*xs) -> tuple`` of output arrays. ``xs`` may be arrays from *any*
        framework; the source adapter converts them to its own type. Outputs are
        returned in the source framework's native array type.
    spec:
        Optional ``spec(*in_specs) -> tuple[Spec]``. If omitted, output specs are
        inferred by evaluating once on zeros (cached per input signature).
    vjp:
        Optional zero-argument factory returning the reverse-mode rule as a
        DiffFunction ``g(*xs, *cts) -> input cotangents``. ``None`` means the
        function is not differentiable beyond its current order.
    jvp:
        Optional zero-argument factory returning the forward-mode rule
        ``h(*xs, *tangents) -> output tangents``.
    fwd / bwd:
        Optional residual-caching fast path for eager first-order use:
        ``fwd(*xs) -> (outputs, residuals)`` and ``bwd(residuals, *cts) -> input cts``.
    batched:
        True if ``eval`` natively accepts one extra leading batch dimension on
        every input (used to vectorize vmap instead of looping).
    framework / name / order:
        Metadata for error messages; ``order`` is the derivative order this
        function represents (0 = original function).
    """

    def __init__(
        self,
        eval: Callable[..., Sequence[Any]],
        *,
        spec: Optional[Callable[..., Sequence[Spec]]] = None,
        vjp: Optional[Callable[[], "DiffFunction"]] = None,
        jvp: Optional[Callable[[], "DiffFunction"]] = None,
        fwd: Optional[Callable[..., tuple]] = None,
        bwd: Optional[Callable[..., Sequence[Any]]] = None,
        batched: bool = False,
        framework: str = "unknown",
        name: Optional[str] = None,
        order: int = 0,
        max_order_reason: Optional[str] = None,
        n_inputs: Optional[int] = None,
        n_outputs: Optional[int] = None,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._eval = eval
        self._spec = spec
        self._vjp_factory = vjp
        self._jvp_factory = jvp
        self._fwd = fwd
        self._bwd = bwd
        self.batched = batched
        self.framework = framework
        self.name = name or f"fn{next(_name_counter)}"
        self.order = order
        self.max_order_reason = max_order_reason
        self._spec_cache: dict = {}
        self._lock = threading.Lock()
        self._vjp_fn: Optional[DiffFunction] = None
        self._jvp_fn: Optional[DiffFunction] = None
        # Set by the source adapters: the original function, so a consumer of the same
        # framework composes it natively instead of bridging (no callback, no nested autodiff).
        self._jax_native: Optional[Callable] = None
        self._torch_native: Optional[Callable] = None
        self._warp_native: Optional[Callable] = None
        self._tf_native: Optional[Callable] = None
        # True when evaluating this function has effects beyond its outputs (a PyTorch VJP
        # accumulates into captured parameters' .grad): compilers must not drop or repeat it.
        self.has_side_effects = False
        # Consumers (to_torch, to_jax, ...) cache their wrappers here.
        self._consumer_cache: dict = {}

    # ------------------------------------------------------------------ eval
    def eval(self, *xs) -> tuple:
        """Evaluate the function. Returns a tuple of source-native arrays."""
        outs = _as_tuple(self._eval(*xs))
        # Learn arities on first use; derivative rules rely on them to split
        # ``(*primals, *cotangents)`` argument lists at any order.
        if self.n_inputs is None:
            self.n_inputs = len(xs)
        if self.n_outputs is None:
            self.n_outputs = len(outs)
        return outs

    __call__ = eval

    # ------------------------------------------------------------------ spec
    def spec(self, *in_specs: Spec) -> tuple:
        """Output specs for the given input specs (shapes/dtypes only)."""
        key = tuple((s.shape, s.dtype.str) for s in in_specs)
        with self._lock:
            if key in self._spec_cache:
                return self._spec_cache[key]
        if self._spec is not None:
            out = tuple(self._spec(*in_specs))
        else:
            from . import _convert

            outs = self.eval(*[s.zeros() for s in in_specs])
            out = tuple(Spec(np.shape(a), _convert.numpy_dtype(a)) for a in (_convert.to_numpy(o) for o in outs))
        with self._lock:
            self._spec_cache[key] = out
        if self.n_inputs is None:
            self.n_inputs = len(in_specs)
        if self.n_outputs is None:
            self.n_outputs = len(out)
        return out

    # ------------------------------------------------------------------ derivative rules
    @property
    def has_vjp(self) -> bool:
        return self._vjp_factory is not None

    @property
    def has_jvp(self) -> bool:
        return self._jvp_factory is not None

    @property
    def vjp_function(self) -> "DiffFunction":
        """Reverse-mode rule ``g(*xs, *cts) -> input cotangents`` as a DiffFunction."""
        if self._vjp_factory is None:
            reason = self.max_order_reason or f"{self.framework} source provides no reverse-mode rule"
            raise MaxOrderError(
                f"gradbridge: cannot differentiate '{self.name}' (derivative order {self.order + 1}): {reason}."
            )
        with self._lock:
            if self._vjp_fn is None:
                fn = self._vjp_factory()
                fn.order = self.order + 1
                self._vjp_fn = fn
            return self._vjp_fn

    @property
    def jvp_function(self) -> "DiffFunction":
        """Forward-mode rule ``h(*xs, *tangents) -> output tangents`` as a DiffFunction."""
        if self._jvp_factory is None:
            raise MaxOrderError(
                f"gradbridge: '{self.name}' ({self.framework}) has no forward-mode (JVP) rule."
            )
        with self._lock:
            if self._jvp_fn is None:
                fn = self._jvp_factory()
                fn.order = self.order + 1
                self._jvp_fn = fn
            return self._jvp_fn

    # ------------------------------------------------------------------ residual fast path
    def fwd(self, *xs):
        """Forward pass that also returns residuals for :meth:`bwd`."""
        if self._fwd is not None:
            outs, res = self._fwd(*xs)
            return _as_tuple(outs), res
        return self.eval(*xs), xs

    def bwd(self, residuals, *cts) -> tuple:
        """Reverse pass from residuals produced by :meth:`fwd`."""
        if self._bwd is not None:
            return _as_tuple(self._bwd(residuals, *cts))
        return self.vjp_function.eval(*residuals, *cts)

    def __gradbridge__(self) -> "DiffFunction":
        return self

    def __repr__(self) -> str:
        return f"<gradbridge.DiffFunction {self.name!r} from {self.framework}, order={self.order}>"


def split_primals(fn: DiffFunction, args: Sequence) -> tuple:
    """Split the argument list of ``fn``'s VJP/JVP rule into ``(primals, cotangents)``.

    A VJP rule of ``fn`` is called as ``g(*primals, *cotangents)`` where there is
    one cotangent per output of ``fn`` (one tangent per input for a JVP rule).
    """
    args = tuple(args)
    if fn.n_inputs is not None:
        k = fn.n_inputs
    elif fn.n_outputs is not None:
        k = len(args) - fn.n_outputs
    else:
        raise RuntimeError(
            f"gradbridge: cannot tell how many inputs '{fn.name}' takes; call it once before "
            "differentiating it, or pass n_inputs= when wrapping."
        )
    if not 0 <= k <= len(args):
        raise RuntimeError(f"gradbridge: bad argument count {len(args)} for derivative of '{fn.name}'")
    return args[:k], args[k:]


def infer_n_inputs(fn: Callable, n_inputs: Optional[int]) -> Optional[int]:
    """Count fixed positional parameters of a user function (None if it takes *args)."""
    if n_inputs is not None:
        return n_inputs
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    count = 0
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            return None
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            count += 1
    return count


def as_diff_function(obj) -> DiffFunction:
    """Coerce ``obj`` to a :class:`DiffFunction` via the ``__gradbridge__`` protocol."""
    if isinstance(obj, DiffFunction):
        return obj
    hook = getattr(obj, "__gradbridge__", None)
    if hook is not None:
        fn = hook()
        if not isinstance(fn, DiffFunction):
            raise TypeError(f"{type(obj).__name__}.__gradbridge__() must return a gradbridge.DiffFunction")
        return fn
    raise TypeError(
        f"gradbridge: {obj!r} is not a DiffFunction. Wrap it first with gradbridge.from_torch / "
        "from_jax / from_tf / from_numpy / from_warp, or implement __gradbridge__()."
    )
