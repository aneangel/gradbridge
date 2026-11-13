"""Gradient checking that works the same way for every framework.

``gradcheck`` compares a function's analytic VJP (any order) against central
finite differences. ``compare`` checks that two implementations -- say the same
model in JAX and in PyTorch -- agree on values and gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import _convert, _fd
from ._core import as_diff_function


@dataclass
class InputCheck:
    index: int
    shape: tuple
    max_abs_err: float
    max_rel_err: float
    ok: bool


@dataclass
class GradcheckReport:
    name: str
    order: int
    inputs: List[InputCheck] = field(default_factory=list)
    note: str = ""

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.inputs)

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"gradcheck '{self.name}' (order {self.order}): {status}"]
        if self.note:
            lines.append(f"  {self.note}")
        lines.append(f"  {'input':>5}  {'shape':<14} {'max abs err':>12} {'max rel err':>12}  ok")
        for c in self.inputs:
            lines.append(
                f"  {c.index:>5}  {str(c.shape):<14} {c.max_abs_err:>12.3e} {c.max_rel_err:>12.3e}  {'yes' if c.ok else 'NO'}"
            )
        return "\n".join(lines)


def _np(x):
    return np.asarray(_convert.to_numpy(x))


def _random_like(rng, arrays):
    return [
        rng.standard_normal(np.shape(a)).astype(a.dtype) if np.issubdtype(a.dtype, np.floating) else np.zeros_like(a)
        for a in arrays
    ]


def gradcheck(
    fn,
    *xs,
    order: int = 1,
    eps: Optional[float] = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    seed: int = 0,
    raise_on_fail: bool = False,
) -> GradcheckReport:
    """Check ``fn``'s order-``order`` reverse-mode derivative against finite differences.

    Use float64 inputs: float32 finite differences are too noisy for tight
    tolerances. Order 2 checks the VJP-of-VJP (Hessian-vector products), the
    same way ``torch.autograd.gradgradcheck`` does.
    """
    dfn = as_diff_function(fn)
    rng = np.random.default_rng(seed)
    args = [_np(x) for x in xs]
    target = dfn
    for _ in range(order - 1):
        outs = [_np(o) for o in target.eval(*args)]
        args = args + _random_like(rng, outs)
        target = target.vjp_function

    outs = [_np(o) for o in target.eval(*args)]
    cts = _random_like(rng, outs)
    analytic = [_np(g) for g in target.vjp_function.eval(*args, *cts)]
    numeric = _fd.fd_vjp(target, *args, cts=cts, eps=eps)

    report = GradcheckReport(name=dfn.name, order=order)
    if any(a.dtype == np.float32 for a in args):
        report.note = "float32 inputs: finite differences are noisy, prefer float64"
    for i, (a, n, x) in enumerate(zip(analytic, numeric, args)):
        a = a.astype(np.float64).reshape(np.shape(x))
        n = np.asarray(n, dtype=np.float64)
        abs_err = float(np.max(np.abs(a - n))) if a.size else 0.0
        denom = np.maximum(np.abs(n), atol)
        rel_err = float(np.max(np.abs(a - n) / denom)) if a.size else 0.0
        ok = bool(np.allclose(a, n, rtol=rtol, atol=atol))
        report.inputs.append(InputCheck(i, tuple(np.shape(x)), abs_err, rel_err, ok))
    if raise_on_fail and not report.passed:
        raise AssertionError(str(report))
    return report


@dataclass
class CompareReport:
    name_a: str
    name_b: str
    max_value_err: float
    max_grad_err: float
    passed: bool

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        status = "MATCH" if self.passed else "MISMATCH"
        return (
            f"compare '{self.name_a}' vs '{self.name_b}': {status}  "
            f"(max value err {self.max_value_err:.3e}, max grad err {self.max_grad_err:.3e})"
        )


def compare(fn_a, fn_b, *xs, rtol: float = 1e-5, atol: float = 1e-7, seed: int = 0) -> CompareReport:
    """Check that two implementations agree on outputs and on VJPs at ``xs``."""
    a, b = as_diff_function(fn_a), as_diff_function(fn_b)
    rng = np.random.default_rng(seed)
    args = [_np(x) for x in xs]
    oa = [_np(o) for o in a.eval(*args)]
    ob = [_np(o) for o in b.eval(*args)]
    if len(oa) != len(ob) or any(x.shape != y.shape for x, y in zip(oa, ob)):
        return CompareReport(a.name, b.name, float("inf"), float("inf"), False)
    val_err = max((float(np.max(np.abs(x - y))) if x.size else 0.0) for x, y in zip(oa, ob)) if oa else 0.0
    cts = _random_like(rng, oa)
    ga = [_np(g) for g in a.vjp_function.eval(*args, *cts)]
    gb = [_np(g) for g in b.vjp_function.eval(*args, *cts)]
    grad_err = max((float(np.max(np.abs(x - y))) if np.size(x) else 0.0) for x, y in zip(ga, gb)) if ga else 0.0
    ok = all(np.allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(oa, ob)) and all(
        np.allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(ga, gb)
    )
    return CompareReport(a.name, b.name, val_err, grad_err, ok)
