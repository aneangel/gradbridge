"""NVIDIA Warp adapter.

``from_warp`` turns a Python function that launches Warp kernels into a
DiffFunction. The forward pass is recorded on a ``wp.Tape`` and the backward
pass replays it. Warp's tape computes first-order adjoints only, so asking for
a second derivative raises :class:`gradbridge.MaxOrderError`.

``to_warp`` exposes any DiffFunction to Warp code: call it with Warp arrays
inside a ``wp.Tape`` and ``tape.backward()`` propagates gradients through it
via ``Tape.record_func``.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, Callable, Optional, Sequence

import numpy as np
import warp as wp

from . import _convert
from ._core import DiffFunction, _as_tuple, as_diff_function, infer_n_inputs, split_primals

__all__ = ["from_warp", "to_warp"]

_FIRST_ORDER_ONLY = "Warp's Tape computes first-order adjoints only (no gradient-of-gradient)"


def _active_tape():
    """The currently active ``wp.Tape`` (Warp keeps it in a private runtime slot)."""
    try:
        return wp._src.context.runtime.tape
    except AttributeError:  # pragma: no cover - older/newer Warp layouts
        try:
            import warp.context as ctx

            return ctx.runtime.tape
        except Exception:
            return None


@contextlib.contextmanager
def _tape_paused():
    """Temporarily hide the active ``wp.Tape`` so launches inside are not recorded on it."""
    try:
        runtime = wp._src.context.runtime
    except AttributeError:  # pragma: no cover - older/newer Warp layouts
        import warp.context as ctx

        runtime = ctx.runtime
    tape, runtime.tape = runtime.tape, None
    try:
        yield
    finally:
        runtime.tape = tape


# =============================================================================
# Residual reuse detection
# =============================================================================
# A cached Warp tape replays adjoints that read the forward's arrays. Simulators reuse
# preallocated buffers, so a later call can overwrite what an earlier, not yet backpropagated
# tape depends on. Every recorded forward registers the memory its kernels touched; a later
# Warp call touching any of it marks the earlier residual stale, and its backward recomputes.
# Kernels may write arrays passed as inputs, so all touched memory counts as possibly written.
_LIVE_RESIDUALS: list = []


_UNSET = object()


class _Residual:
    __slots__ = ("tape", "wxs", "outs", "xs", "_ranges", "stale", "__weakref__")

    def __init__(self, tape, wxs, outs, xs):
        self.tape, self.wxs, self.outs, self.xs = tape, wxs, outs, xs
        self._ranges = _UNSET
        self.stale = False

    @property
    def ranges(self):
        # Computed only when an overlap check needs it: in the common forward -> backward pattern
        # no other residual is alive, and a long tape is never scanned.
        if self._ranges is _UNSET:
            self._ranges = _touched_ranges(self.tape)
        return self._ranges


def _is_plain_value(arg) -> bool:
    if arg is None or isinstance(arg, (bool, int, float, np.generic)):
        return True
    return hasattr(type(arg), "_wp_scalar_type_")  # Warp vector/matrix/quaternion/transform values


def _touched_ranges(tape):
    """Merged ``(lo, hi)`` address ranges of every array the tape's launches touched.

    ``None`` when the tape holds something it cannot describe (a recorded Python function, a
    struct argument): such a residual is treated as overlapping everything.
    """
    los, his = [], []
    for launch in tape.launches:
        if not isinstance(launch, list):
            return None
        for arg in (*launch[3], *launch[4]):
            if isinstance(arg, wp.array):
                if arg.ptr:
                    los.append(arg.ptr)
                    his.append(arg.ptr + max(arg.capacity, 1))
            elif not _is_plain_value(arg):
                return None
    if not los:
        return np.empty(0, np.uint64), np.empty(0, np.uint64)
    lo, hi = np.asarray(los, np.uint64), np.asarray(his, np.uint64)
    order = np.argsort(lo)
    lo, hi = lo[order], np.maximum.accumulate(hi[order])
    # Merge overlapping intervals: a new run starts where lo exceeds every earlier hi.
    starts = np.concatenate(([True], lo[1:] >= hi[:-1]))
    run = np.cumsum(starts) - 1
    merged_hi = np.zeros(int(run[-1]) + 1, np.uint64)
    np.maximum.at(merged_hi, run, hi)
    return lo[starts], merged_hi


def _overlaps(a, b) -> bool:
    if a is None or b is None:
        return True
    (alo, ahi), (blo, bhi) = a, b
    if not len(alo) or not len(blo):
        return False
    idx = np.searchsorted(alo, bhi, side="left") - 1  # last interval of a starting before each b ends
    valid = idx >= 0
    return bool(np.any(ahi[idx[valid]] > blo[valid]))


def _invalidate(new, everything=False):
    """Mark live residuals that ``new`` (a _Residual) may have overwritten stale.

    ``everything=True`` marks them all, for calls whose touched memory is unknown.
    """
    alive = []
    for ref in _LIVE_RESIDUALS:
        res = ref()
        if res is None:
            continue
        alive.append(ref)
        if res is not new and not res.stale and (everything or _overlaps(res.ranges, new.ranges)):
            res.stale = True
    _LIVE_RESIDUALS[:] = alive


def _torch_cuda(t) -> bool:
    return _convert.framework_of(t) == "torch" and t.is_cuda


@wp.kernel
def _accumulate(dst: wp.array(dtype=Any), src: wp.array(dtype=Any)):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


# =============================================================================
# Source: Warp program -> DiffFunction
# =============================================================================
def from_warp(
    fn: Callable,
    *,
    in_dtypes: Optional[Sequence] = None,
    device=None,
    n_inputs: Optional[int] = None,
    name: Optional[str] = None,
    batched: bool = False,
) -> DiffFunction:
    """Wrap ``fn(*wp_arrays) -> wp.array | tuple`` that launches Warp kernels.

    ``fn`` must allocate its outputs with ``requires_grad=True`` so the tape can
    hold their adjoints, e.g. ``wp.zeros_like(x, requires_grad=True)``.

    ``in_dtypes`` lets inputs be Warp vector/matrix types: a NumPy/Torch array
    of shape ``(N, 3)`` becomes ``wp.array(dtype=wp.vec3, shape=N)``.
    """
    name = name or getattr(fn, "__name__", "warp_fn")
    n_in = infer_n_inputs(fn, n_inputs)

    def _inputs(xs, requires_grad):
        out = []
        for i, x in enumerate(xs):
            dt = in_dtypes[i] if in_dtypes is not None and i < len(in_dtypes) else None
            rg = requires_grad and (dt is not None or _convert.is_float(x))
            out.append(_convert.to_warp(x, dtype=dt, requires_grad=rg, device=device))
        return out

    def _eval(*xs):
        # No tape, so the touched memory is unknown: conservatively invalidate every cached tape.
        _invalidate(None, everything=True)
        return _as_tuple(fn(*_inputs(xs, requires_grad=False)))

    def _fwd(*xs):
        wxs = _inputs(xs, requires_grad=True)
        tape = wp.Tape()
        with tape:
            outs = _as_tuple(fn(*wxs))
        for o in outs:
            if o.grad is None:
                raise RuntimeError(
                    f"gradbridge.from_warp('{name}'): outputs must be allocated with requires_grad=True "
                    "so Warp can store their adjoints."
                )
        res = _Residual(tape, wxs, outs, xs)
        _invalidate(res)
        _LIVE_RESIDUALS.append(weakref.ref(res))
        return outs, res

    def _bwd(res, *cts):
        if res.stale:
            # A later call overwrote memory this tape reads: replay the forward first.
            res = _fwd(*res.xs)[1]
        tape, wxs, outs = res.tape, res.wxs, res.outs
        # Outputs always own a .grad here, so Tape.backward copies the cotangents into it
        # instead of adopting the caller's arrays as .grad (which tape.zero() would then wipe).
        grads = {o: _convert.to_warp(c, dtype=o.dtype, device=o.device) for o, c in zip(outs, cts)}
        tape.zero()
        tape.backward(grads=grads)
        # Clone: recorded adjoint launches keep pointing at these buffers, so a reuse of this
        # residual (retain_graph) writes into them again. No trailing tape.zero(): every backward
        # zeroes first, so clearing again here would only cost memsets.
        return tuple(
            wp.clone(x.grad) if x.requires_grad and x.grad is not None else wp.zeros_like(x) for x in wxs
        )

    dfn = DiffFunction(
        _eval,
        fwd=_fwd,
        bwd=_bwd,
        batched=batched,
        framework="warp",
        name=name,
        n_inputs=n_in,
    )

    def _vjp_factory():
        def g_eval(*args):
            xs, cts = split_primals(dfn, args)
            outs, res = _fwd(*xs)
            return _bwd(res, *cts)

        return DiffFunction(
            g_eval,
            framework="warp",
            name=f"{name}_vjp",
            n_outputs=dfn.n_inputs,
            max_order_reason=_FIRST_ORDER_ONLY,
        )

    dfn._vjp_factory = _vjp_factory
    if in_dtypes is None and device is None:
        dfn._warp_native = fn
    return dfn


# =============================================================================
# Consumer: DiffFunction -> Warp-callable
# =============================================================================
def to_warp(fn, *, out_dtypes: Optional[Sequence] = None, device=None) -> Callable:
    """Expose any gradbridge function to Warp code.

    Call the result with Warp arrays. If a ``wp.Tape`` is active and any input
    has ``requires_grad=True``, the call is recorded so ``tape.backward()``
    flows gradients through it. Torch, JAX and Warp sources receive the arrays
    zero-copy; TensorFlow and NumPy sources go through host memory.
    """
    dfn = as_diff_function(fn)
    native = dfn._warp_native if out_dtypes is None and device is None else None

    def wrapper(*wxs):
        if native is not None and all(isinstance(x, wp.array) for x in wxs):
            # A Warp source called from Warp needs no bridge: its kernels record on the caller's tape.
            outs = _as_tuple(native(*wxs))
            return outs[0] if len(outs) == 1 else outs
        # The bridged function is opaque to the caller's tape: its own kernels must not be
        # recorded there, or tape.backward() would count its gradient twice.
        need_grad = any(getattr(x, "requires_grad", False) for x in wxs)
        tape = _active_tape()
        record = tape is not None and need_grad
        with _tape_paused():
            # Recording: keep the source's residuals for the backward instead of recomputing.
            if record:
                outs, res = dfn.fwd(*wxs)
            else:
                outs = dfn.eval(*wxs)
        dev = device or next((x.device for x in wxs if isinstance(x, wp.array)), None)
        w_outs = []
        for i, o in enumerate(outs):
            dt = out_dtypes[i] if out_dtypes is not None and i < len(out_dtypes) else None
            rg = need_grad and _convert.is_float(o)
            w_outs.append(_convert.to_warp(o, dtype=dt, requires_grad=rg, device=dev))

        if record:
            grad_inputs = [x for x in wxs if isinstance(x, wp.array) and x.requires_grad]

            def backward():
                cts = [o.grad if o.grad is not None else wp.zeros_like(o) for o in w_outs]
                with _tape_paused():
                    in_cts = dfn.bwd(res, *cts)
                pairs = [(x, ct) for x, ct in zip(wxs, in_cts) if isinstance(x, wp.array) and x.requires_grad]
                if pairs and all(_torch_cuda(ct) and x.grad.device.is_cuda for x, ct in pairs):
                    # PyTorch cotangents: accumulate with in-place torch adds on zero-copy views of
                    # x.grad (one stream handshake each way for all of them) instead of a Warp
                    # launch plus a conversion per input.
                    device = pairs[0][1].device
                    _convert._torch_after_warp(device)
                    for x, ct in pairs:
                        view = wp.to_torch(x.grad, requires_grad=False)
                        view.add_(ct.reshape(view.shape).to(view.dtype))
                    _convert._warp_after_torch(device)
                    return
                for x, ct in pairs:
                    g = _convert.to_warp(ct, dtype=x.grad.dtype, device=x.grad.device)
                    wp.launch(
                        _accumulate,
                        dim=x.grad.size,
                        inputs=[x.grad.flatten(), g.flatten()],
                        device=x.grad.device,
                    )

            tape.record_func(backward, arrays=grad_inputs + [o for o in w_outs if o.requires_grad])

        return w_outs[0] if len(w_outs) == 1 else tuple(w_outs)

    wrapper.__name__ = dfn.name
    wrapper.__gradbridge__ = lambda: dfn
    return wrapper
