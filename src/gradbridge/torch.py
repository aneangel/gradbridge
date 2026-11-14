"""PyTorch adapter.

``from_torch`` turns a PyTorch function into a DiffFunction whose derivative
rules are built with ``torch.autograd.grad(create_graph=True)``, so any order
of reverse-mode derivative is available.

``to_torch`` exposes any DiffFunction as a ``torch.library.custom_op`` (with a
fake-tensor kernel and vmap rule) wrapped in a ``torch.autograd.Function``
(with ``setup_context`` and a generated vmap rule). That makes it work with
``loss.backward()``, double backward (``create_graph=True``), ``torch.func.grad``
/ ``torch.func.vmap`` and ``torch.compile(fullgraph=True)``.
Forward mode uses the source's JVP rule for ``torch.func.jvp`` / ``jacfwd``.
"""

from __future__ import annotations

import functools
import importlib.util
import itertools
import re
import sys
import threading
from typing import Callable, Optional

# torch.library custom ops lazily import torch._dynamo, which loads Triton's
# bundled LLVM. If TensorFlow (which bundles its own LLVM) is already loaded,
# some binary combinations abort the whole process. Load it now, or keep the
# conservative import guard until those combinations have a compatibility matrix.
if (
    "tensorflow" in sys.modules
    and "torch._dynamo" not in sys.modules
    and importlib.util.find_spec("triton") is not None
):
    raise ImportError(
        "gradbridge.torch must be imported before TensorFlow when Triton is installed: "
        "Some TensorFlow and Triton binaries bundle conflicting LLVM builds; this import guard is conservative. "
        "Import gradbridge.torch (or torch._dynamo) first, then tensorflow."
    )

import torch
import torch._dynamo  # noqa: F401  (see note above)
from torch._functorch import eager_transforms as _transforms

from . import _convert
from ._core import DiffFunction, MaxOrderError, Spec, _as_tuple, as_diff_function, infer_n_inputs, split_primals

__all__ = ["from_torch", "to_torch"]

_op_counter = itertools.count()


# =============================================================================
# Source: PyTorch function -> DiffFunction
# =============================================================================
# Custom-op bodies run *below* PyTorch's autograd layer (autograd dispatch keys
# are excluded), so a PyTorch source called from inside one -- directly, or via
# a JAX/TF callback -- could not record a graph. Snapshot the normal dispatch
# state at import time and restore it while a PyTorch source differentiates.
_DEFAULT_INCLUDE = torch._C._dispatch_tls_local_include_set()
_DEFAULT_EXCLUDE = torch._C._dispatch_tls_local_exclude_set()


class _autograd_on:
    def __enter__(self):
        self._force = torch._C._ForceDispatchKeyGuard(_DEFAULT_INCLUDE, _DEFAULT_EXCLUDE)
        self._force.__enter__()
        self._grad = torch.enable_grad()
        self._grad.__enter__()
        return self

    def __exit__(self, *exc):
        self._grad.__exit__(*exc)
        self._force.__exit__(*exc)
        return False


def _leaf(x: torch.Tensor) -> torch.Tensor:
    """Make ``x`` something we can differentiate with respect to, without
    detaching it from an enclosing graph (needed for higher-order derivatives)."""
    if not (x.is_floating_point() or x.is_complex()):
        return x
    if x.requires_grad:
        return x
    return x.detach().requires_grad_(True)


# Set while a function is evaluated only to infer output shapes: that dry run must not leave
# gradients in captured parameters.
_SHAPE_ONLY = threading.local()


def _captured_leaves(outs, xs):
    """Leaf tensors that require grad and that ``outs`` depend on, other than the inputs ``xs``:
    typically an ``nn.Module``'s parameters captured by the wrapped function.

    The graph walk stops at the inputs (and at their own graph nodes, for non-leaf inputs), so a
    caller's graph upstream of the inputs is never reached.
    """
    inputs = {id(x) for x in xs}
    stop = {id(x.grad_fn) for x in xs if isinstance(x, torch.Tensor) and x.grad_fn is not None}
    stack = [o.grad_fn for o in outs if isinstance(o, torch.Tensor) and o.grad_fn is not None]
    # Visited nodes are kept alive: graph nodes are transient Python wrappers, and a freed
    # wrapper's id() can be reused by the next one, which would then be wrongly skipped.
    seen, visited, leaves, leaf_ids = set(), [], [], set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen or id(node) in stop:
            continue
        seen.add(id(node))
        visited.append(node)
        var = getattr(node, "variable", None)  # AccumulateGrad nodes hold their leaf
        if var is not None:
            if id(var) not in inputs and id(var) not in leaf_ids:
                leaf_ids.add(id(var))
                leaves.append(var)
            continue
        stack.extend(nxt for nxt, _ in node.next_functions)
    return leaves


def _grads(outs, xs, cts, *, create_graph: bool, owned: bool = False):
    """VJP of ``outs`` w.r.t. ``xs`` with cotangents ``cts``; zeros where unused.

    Like ``backward()`` in PyTorch itself, it also accumulates into ``.grad`` of captured leaves
    (e.g. a network's parameters), so they train when a non-PyTorch framework drives the call.
    ``owned``: every input is a fresh leaf created by gradbridge, so ``torch.autograd.backward``
    can accumulate into all leaves natively (no graph walk, no Python loop over parameters).
    """
    idx = [i for i, x in enumerate(xs) if x.requires_grad]
    pairs = [(o, c) for o, c in zip(outs, cts) if isinstance(o, torch.Tensor) and o.requires_grad]
    shape_only = getattr(_SHAPE_ONLY, "active", False)
    if owned and not create_graph and not shape_only and pairs:
        for x in xs:
            x.grad = None  # residuals may be reused (retain_graph): start from zero
        torch.autograd.backward(
            [o for o, _ in pairs],
            grad_tensors=[c.to(dtype=o.dtype, device=o.device) for o, c in pairs],
            retain_graph=True,
        )
        return tuple(x.grad if x.grad is not None else torch.zeros_like(x) for x in xs)
    result = [torch.zeros_like(x) for x in xs]
    captured = _captured_leaves([o for o, _ in pairs], xs) if pairs and not shape_only else []
    if (idx or captured) and pairs:
        g = torch.autograd.grad(
            [o for o, _ in pairs],
            [xs[i] for i in idx] + captured,
            [c.to(dtype=o.dtype, device=o.device) for o, c in pairs],
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )
        for i, gi in zip(idx, g[: len(idx)]):
            if gi is not None:
                result[i] = gi
        with torch.no_grad():
            for leaf, gl in zip(captured, g[len(idx) :]):
                if gl is not None:
                    leaf.grad = gl.detach().clone() if leaf.grad is None else leaf.grad.add_(gl.detach())
    return tuple(result)


def _make(fn_t: Callable, *, name, framework="torch", n_inputs=None, n_outputs=None, batched=False, device=None):
    def to_t(x):
        return _convert.to_torch(x, device=device)

    def _eval(*xs):
        with torch.no_grad():
            return tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in _as_tuple(fn_t(*[to_t(x) for x in xs])))

    def _fwd(*xs):
        # Fresh leaves: the residual path is first-order only, so never attach
        # to the caller's graph.
        with _autograd_on():
            txs = [_leaf(to_t(x).detach()) for x in xs]
            outs = _as_tuple(fn_t(*txs))
        return tuple(o.detach() for o in outs), (txs, outs)

    def _bwd(res, *cts):
        txs, outs = res
        with _autograd_on():
            # txs are fresh leaves made by _fwd: PyTorch can accumulate into every leaf natively.
            return _grads(outs, txs, [to_t(c) for c in cts], create_graph=False, owned=True)

    def _spec(*specs):
        _SHAPE_ONLY.active = True
        try:
            return _infer_spec(*specs)
        finally:
            _SHAPE_ONLY.active = False

    def _infer_spec(*specs):
        try:
            metas = [torch.empty(s.shape, dtype=_convert.torch_dtype(s.dtype), device="meta") for s in specs]
            with torch.no_grad():
                outs = _as_tuple(fn_t(*metas))
            return tuple(Spec(tuple(o.shape), _convert.numpy_dtype_from_torch(o.dtype)) for o in outs)
        except Exception:
            # Meta tensors fail when the function captures real tensors (e.g. a CUDA module):
            # evaluate on zeros, on the device those captured tensors most likely live on.
            devices = [device] if device is not None else []
            devices += (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
            last = None
            for dev in devices:
                try:
                    with torch.no_grad():
                        zs = [torch.zeros(s.shape, dtype=_convert.torch_dtype(s.dtype), device=dev) for s in specs]
                        outs = _as_tuple(fn_t(*zs))
                    return tuple(Spec(tuple(o.shape), _convert.numpy_dtype_from_torch(o.dtype)) for o in outs)
                except Exception as e:  # noqa: PERF203 - try the next device
                    last = e
            raise last

    dfn = DiffFunction(
        _eval,
        spec=_spec,
        fwd=_fwd,
        bwd=_bwd,
        batched=batched,
        framework=framework,
        name=name,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )

    def vjp_body(*args):
        xs, cts = split_primals(dfn, args)
        # Grad mode on entry tells the callers apart: evaluated for a foreign consumer it is off
        # (outputs are detached), inside the higher-order machinery it is on and a graph is needed.
        need_graph = torch.is_grad_enabled()
        with _autograd_on():
            converted = [to_t(x) for x in xs]
            txs = [_leaf(c) for c in converted]
            owned = all(t is not c for t, c in zip(txs, converted) if t.requires_grad)
            tcts = [to_t(c) for c in cts]
            outs = _as_tuple(fn_t(*txs))
            return _grads(outs, txs, tcts, create_graph=need_graph, owned=owned)

    def jvp_body(*args):
        k = len(args) // 2
        xs = tuple(to_t(x) for x in args[:k])
        ts = tuple(to_t(t).to(x.dtype) for x, t in zip(xs, args[k:]))
        with _autograd_on():
            _, tangents_out = torch.func.jvp(lambda *a: _as_tuple(fn_t(*a)), xs, ts)
        return tangents_out

    def _vjp_factory():
        vjp = _make(
            vjp_body, name=f"{name}_vjp", framework=framework, n_outputs=dfn.n_inputs,
            batched=batched, device=device,
        )
        vjp.has_side_effects = True  # accumulates into captured parameters' .grad
        return vjp

    dfn._vjp_factory = _vjp_factory
    dfn._jvp_factory = lambda: _make(jvp_body, name=f"{name}_jvp", framework=framework, batched=batched, device=device)
    return dfn


def from_torch(
    fn: Callable,
    *,
    batched: bool = False,
    n_inputs: Optional[int] = None,
    name: Optional[str] = None,
    device=None,
) -> DiffFunction:
    """Wrap a PyTorch function ``fn(*tensors) -> tensor | tuple``.

    Gradients of any order come from PyTorch's own autograd. ``device`` moves
    incoming arrays to a torch device (e.g. ``"cuda"``) before calling ``fn``.
    """
    dfn = _make(
        fn,
        name=name or getattr(fn, "__name__", "torch_fn"),
        n_inputs=infer_n_inputs(fn, n_inputs),
        batched=batched,
        device=device,
    )
    if device is None:
        dfn._torch_native = fn
    return dfn


# =============================================================================
# Consumer: DiffFunction -> PyTorch op
# =============================================================================
def _spec_of(x: torch.Tensor) -> Spec:
    return Spec(tuple(int(d) for d in x.shape), _convert.numpy_dtype_from_torch(x.dtype))


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    try:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() and a.numel() > 0
    except Exception:
        return False


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return s if s and not s[0].isdigit() else f"f_{s}"


def _custom_op(dfn: DiffFunction):
    cache = dfn._consumer_cache
    if "torch_op" in cache:
        return cache["torch_op"]

    opname = f"gradbridge::{_sanitize(dfn.name)}_{next(_op_counter)}"

    @torch.library.custom_op(opname, mutates_args=())
    def op(xs: list[torch.Tensor]) -> list[torch.Tensor]:
        device = xs[0].device if xs else torch.device("cpu")
        outs = dfn.eval(*xs)
        specs = dfn.spec(*[_spec_of(x) for x in xs])
        result = []
        for o, s in zip(outs, specs):
            t = _convert.to_torch(o, device=device, dtype=_convert.torch_dtype(s.dtype))
            if any(_shares_storage(t, x) for x in xs):
                t = t.clone()
            result.append(t.contiguous())
        return result

    @op.register_fake
    def _(xs):
        device = xs[0].device if xs else torch.device("cpu")
        specs = dfn.spec(*[_spec_of(x) for x in xs])
        return [torch.empty(s.shape, dtype=_convert.torch_dtype(s.dtype), device=device) for s in specs]


    def setup_context(ctx, inputs, output):
        (xs,) = inputs
        ctx.save_for_backward(*xs)
        ctx.out_meta = [(tuple(o.shape), o.dtype, o.device) for o in output]

    def backward(ctx, grads):
        xs = ctx.saved_tensors
        cts = [
            g if g is not None else torch.zeros(shape, dtype=dtype, device=device)
            for g, (shape, dtype, device) in zip(grads, ctx.out_meta)
        ]
        in_cts = _custom_op(dfn.vjp_function)(list(xs) + list(cts))
        return [
            ct.to(x.dtype) if (x.is_floating_point() or x.is_complex()) else None
            for ct, x in zip(in_cts, xs)
        ]

    op.register_autograd(backward, setup_context=setup_context)

    def vmap_rule(info, in_dims, xs):
        dims = in_dims[0]
        B = info.batch_size
        if dfn.batched:
            xb = [
                x.movedim(d, 0) if d is not None else x.unsqueeze(0).expand(B, *x.shape).contiguous()
                for x, d in zip(xs, dims)
            ]
            outs = op(xb)
            return outs, [0] * len(outs)
        per = [op([x.select(d, b) if d is not None else x for x, d in zip(xs, dims)]) for b in range(B)]
        outs = [torch.stack([p[j] for p in per]) for j in range(len(per[0]))]
        return outs, [0] * len(outs)

    torch.library.register_vmap(op, vmap_rule)
    # Dynamo traces the registered OpOverload handle, not the CustomOpDef object.
    op.overload = getattr(torch.ops.gradbridge, opname.split("::", 1)[1]).default
    cache["torch_op"] = op
    return op


_BRIDGE_REGISTRY: dict = {}


def _eager_vjp(key, xs, cts):
    return _bridge(_BRIDGE_REGISTRY[key].vjp_function, forward_mode=True).apply(*xs, *cts)


def _eager_jvp(dfn, xs, tangents, out_meta):
    txs = [x if isinstance(x, torch.Tensor) else _convert.to_torch(x) for x in xs]
    ts = [t if t is not None else torch.zeros_like(x) for x, t in zip(txs, tangents)]
    outs = _bridge(dfn.jvp_function, forward_mode=True).apply(*txs, *ts)
    return tuple(
        t.to(dtype=dtype, device=device) if dtype.is_floating_point or dtype.is_complex else None
        for t, (_, dtype, device) in zip(outs, out_meta)
    )


def _bridge(dfn: DiffFunction, *, forward_mode=False):
    """autograd.Function around the custom op.

    The custom op is the opaque compute (fake-tensor + vmap rules, so
    torch.compile and vmap see through nothing). Differentiation lives here, in
    an autograd.Function with ``setup_context`` and a generated vmap rule, which
    is what ``torch.func.grad`` / ``vmap(grad(...))`` and double backward need.
    """
    cache = dfn._consumer_cache
    cache_key = "torch_forward_bridge" if forward_mode else "torch_bridge"
    if cache_key in cache:
        return cache[cache_key]
    # The autograd.Function's methods must not close over Python objects Dynamo
    # cannot wrap (like the DiffFunction): use an int key + module registry.
    key = id(dfn)
    _BRIDGE_REGISTRY[key] = dfn
    name = dfn.name
    overload = _custom_op(dfn).overload
    # Pre-create the VJP op so a compiled backward can call it as an opaque op
    # (Dynamo cannot trace the Python lookup of dfn.vjp_function).
    vjp_overload = _custom_op(dfn.vjp_function).overload if dfn.has_vjp else None

    class _Bridge(torch.autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(*xs):
            return tuple(overload(list(xs)))

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)
            ctx.out_meta = [(tuple(o.shape), o.dtype, o.device) for o in output]

        @staticmethod
        def backward(ctx, *grads):
            xs = ctx.saved_tensors
            cts = [
                g if g is not None else torch.zeros(shape, dtype=dtype, device=device)
                for g, (shape, dtype, device) in zip(grads, ctx.out_meta)
            ]
            if torch.compiler.is_compiling():
                if vjp_overload is None:
                    raise MaxOrderError(f"gradbridge: '{name}' has no reverse-mode rule")
                in_cts = vjp_overload(list(xs) + list(cts))
            else:
                # Full path: supports torch.func transforms and double backward.
                # Raises MaxOrderError past the source's limit (e.g. Warp at order 2).
                in_cts = _eager_vjp(key, xs, cts)
            return tuple(
                c.to(x.dtype) if (x.is_floating_point() or x.is_complex()) else None for c, x in zip(in_cts, xs)
            )

    if forward_mode:
        # Dynamo rejects autograd.Function classes with a custom JVP, even when only
        # reverse mode is requested. Keep that method on the eager/torch.func class.
        class _ForwardBridge(_Bridge):
            @staticmethod
            def setup_context(ctx, inputs, output):
                _Bridge.setup_context(ctx, inputs, output)
                ctx.save_for_forward(*inputs)

            @staticmethod
            def jvp(ctx, *tangents):
                return _eager_jvp(_BRIDGE_REGISTRY[key], ctx.saved_tensors, tangents, ctx.out_meta)

        bridge = _ForwardBridge
    else:
        bridge = _Bridge
    bridge.__name__ = f"Bridge_{_sanitize(name)}"
    cache[cache_key] = bridge
    return bridge


def _eager_function(dfn: DiffFunction, cache_residuals: bool):
    """Eager fast path: an autograd.Function that calls the source directly.

    Skips the custom-op dispatcher (needed only by torch.compile / torch.func). With
    ``cache_residuals`` the forward keeps the source's residuals (its autograd graph,
    ``jax.vjp`` pullback, Warp tape) for backward; otherwise backward recomputes them.
    """
    key = "torch_eager_cached" if cache_residuals else "torch_eager_recompute"
    cache = dfn._consumer_cache
    if key in cache:
        return cache[key]

    class _Eager(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *xs):
            # Non-tensor arrays (e.g. a Warp array of constants) pass through unconverted: they
            # are constants to autograd, and the source reads them in its own framework.
            tensors = [x for x in xs if isinstance(x, torch.Tensor)]
            device = tensors[0].device if tensors else torch.device("cpu")
            if cache_residuals:
                outs, ctx.res = dfn.fwd(*xs)
            else:
                outs, ctx.res = dfn.eval(*xs), None
            ctx.save_for_backward(*tensors)
            ctx.save_for_forward(*tensors)
            ctx.constants = [None if isinstance(x, torch.Tensor) else x for x in xs]
            out_t = []
            for o in outs:
                t = _convert.to_torch(o, device=device)
                # Same rule as the custom op: never hand out an alias of an input.
                if any(_shares_storage(t, x) for x in tensors):
                    t = t.clone()
                out_t.append(t)
            ctx.out_meta = [(tuple(o.shape), o.dtype, o.device) for o in out_t]
            return tuple(out_t)

        @staticmethod
        def jvp(ctx, *tangents):
            saved = iter(ctx.saved_tensors)
            xs = [next(saved) if c is None else c for c in ctx.constants]
            return _eager_jvp(dfn, xs, tangents, ctx.out_meta)

        @staticmethod
        def backward(ctx, *grads):
            saved = iter(ctx.saved_tensors)
            xs = [next(saved) if c is None else c for c in ctx.constants]
            cts = [
                g if g is not None else torch.zeros(shape, dtype=dtype, device=device)
                for g, (shape, dtype, device) in zip(grads, ctx.out_meta)
            ]
            if torch.is_grad_enabled():
                # create_graph=True: need a differentiable backward, use the op path.
                txs = [x if isinstance(x, torch.Tensor) else _convert.to_torch(x) for x in xs]
                in_cts = _bridge(dfn.vjp_function, forward_mode=True).apply(*txs, *cts)
            else:
                raw = dfn.bwd(ctx.res, *cts) if ctx.res is not None else dfn.vjp_function.eval(*xs, *cts)
                in_cts = [_convert.to_torch(c, device=x.device) if isinstance(x, torch.Tensor) else None for c, x in zip(raw, xs)]
            return tuple(
                c.to(x.dtype) if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex()) else None
                for c, x in zip(in_cts, xs)
            )

    _Eager.__name__ = f"Eager_{_sanitize(dfn.name)}"
    cache[key] = _Eager
    return _Eager


def _eager_ok() -> bool:
    """Plain eager execution: no torch.compile tracing and no torch.func transform active."""
    if torch.compiler.is_compiling():
        return False
    return torch._C._functorch.peek_interpreter_stack() is None


# Sources whose cached residuals stay valid until backward: PyTorch saved tensors carry version
# counters, JAX and TensorFlow arrays are immutable, NumPy residuals are the inputs themselves.
# Warp kernels write into caller-owned buffers, but from_warp tracks the memory every recorded
# tape touched and recomputes a residual that a later Warp call may have overwritten.
_SAFE_TO_CACHE = {"torch", "jax", "tensorflow", "numpy", "warp"}


def to_torch(fn, *, cache_residuals: Optional[bool] = None) -> Callable:
    """Expose any gradbridge function to PyTorch.

    The result accepts tensors (or any arrays) and returns a tensor, or a tuple
    for multiple outputs. Gradients flow through ``.backward()``.

    In plain eager mode the call goes straight to the source; under
    ``torch.compile`` or a ``torch.func`` transform it goes through a custom op
    those systems can trace. Forward mode (``torch.func.jvp``, ``jacfwd`` and
    ``torch.autograd.forward_ad``) uses the source's JVP rule, when available.
    Compile the complete JVP / forward-Jacobian transform to call the source's
    JVP as a custom op. Nested compiled JVPs are not supported.

    ``cache_residuals`` (eager mode only) controls whether the source's forward
    state is kept for backward instead of recomputing the forward. The default,
    ``None``, caches for every source. For Warp sources, gradbridge records the
    memory each forward's kernels touched; if a later bridged Warp call touches
    any of it before ``backward()`` (e.g. several rollouts of one simulator), the
    earlier backward recomputes its forward first, so gradients stay exact. What
    it cannot see is your own code writing into those buffers between forward and
    backward -- call ``invalidate_warp_residuals`` after external writes, or pass
    ``False`` to always recompute when those writes cannot be annotated.
    Calls without reverse-mode gradients (including ``torch.no_grad()``) skip
    building the source's backward state.
    """
    dfn = as_diff_function(fn)
    op = _custom_op(dfn)
    primal_overload = op.overload
    jvp_overload = _custom_op(dfn.jvp_function).overload if dfn.has_jvp else None
    bridge = _bridge(dfn)
    forward_bridge = _bridge(dfn, forward_mode=True)
    cache = dfn.framework in _SAFE_TO_CACHE if cache_residuals is None else bool(cache_residuals)
    eager = _eager_function(dfn, cache)
    eval_only = _eager_function(dfn, False)

    native = dfn._torch_native
    name = dfn.name

    def wrapper(*xs):
        if native is None and _eager_ok():
            # Eager: arrays from other frameworks are constants to autograd; skip the round trip.
            needs_grad = torch.is_grad_enabled() and any(isinstance(x, torch.Tensor) and x.requires_grad for x in xs)
            outs = tuple((eager if needs_grad else eval_only).apply(*xs))
            return outs[0] if len(outs) == 1 else outs
        txs = [x if isinstance(x, torch.Tensor) else _convert.to_torch(x) for x in xs]
        if native is not None:
            # A PyTorch source called from PyTorch needs no bridge: its own graph is the gradient.
            outs = _as_tuple(native(*txs))
        elif torch.compiler.is_compiling():
            # A custom autograd.Function with a JVP is rejected by Dynamo. Unpack the
            # duals explicitly so the graph contains ordinary primal and tangent ops.
            duals = [torch.autograd.forward_ad.unpack_dual(x) for x in txs]
            if any(t is not None for _, t in duals):
                if _transforms.JVP_NESTING > 1:
                    raise MaxOrderError("gradbridge: compiled nested forward-mode transforms are not supported")
                if jvp_overload is None:
                    raise MaxOrderError(f"gradbridge: '{name}' has no forward-mode (JVP) rule")
                primals = [p for p, _ in duals]
                tangents = [t if t is not None else torch.zeros_like(p) for p, t in duals]
                prim_outs = primal_overload(primals)
                t_outs = jvp_overload(primals + tangents)
                outs = tuple(
                    torch.autograd.forward_ad.make_dual(o, t.to(dtype=o.dtype, device=o.device))
                    if o.is_floating_point() or o.is_complex()
                    else o
                    for o, t in zip(prim_outs, t_outs)
                )
            else:
                outs = tuple(bridge.apply(*txs))
        else:
            outs = tuple(forward_bridge.apply(*txs))
        return outs[0] if len(outs) == 1 else outs

    functools.update_wrapper(wrapper, dfn._eval, assigned=("__doc__",), updated=())
    wrapper.__name__ = dfn.name
    wrapper.__gradbridge__ = lambda: dfn
    wrapper.custom_op = op
    return wrapper


__all__ += ["MaxOrderError"]
