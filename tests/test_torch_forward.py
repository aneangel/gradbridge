"""PyTorch forward-mode transforms and calls that do not need backward state."""

import contextlib

import numpy as np
import pytest
import torch

import gradbridge as gb

from conftest import make_source, ref_values


@pytest.mark.parametrize("source", ["numpy", "torch", "jax", "tf"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_forward_mode(source, dtype, device, data):
    if device == "cuda":
        from test_gpu import require_gpu

        require_gpu("torch", source)
    f = gb.to_torch(make_source(source))
    x, y, tx, ty = (torch.tensor(data[k], dtype=dtype, device=device) for k in ("x", "y", "v", "c1"))
    outs, tangents = torch.func.jvp(f, (x, y), (tx, ty))
    refs = ref_values(x.cpu().numpy(), y.cpu().numpy())
    expected = (torch.cos(x) * y * tx + torch.sin(x) * ty, (2 * x * tx).sum().reshape(1))
    for out, tangent, ref, want in zip(outs, tangents, refs, expected):
        assert out.dtype == tangent.dtype == dtype
        assert out.device == tangent.device == x.device
        np.testing.assert_allclose(out.cpu().numpy(), ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(tangent, want, rtol=1e-5, atol=1e-6)

    # jacfwd batches JVPs, including the zero tangent for the closed-over y.
    jac = torch.func.jacfwd(lambda x: f(x, y)[0])(x)
    torch.testing.assert_close(jac, torch.diag(torch.cos(x) * y), rtol=1e-5, atol=1e-6)
    xs, vs = torch.stack([x, x + 0.1]), torch.stack([tx, tx * 2])
    per_example = torch.func.vmap(lambda x, v: torch.func.jvp(lambda x: f(x, y)[0], (x,), (v,))[1])(xs, vs)
    torch.testing.assert_close(per_example, torch.cos(xs) * y * vs, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("source", ["jax", "tf"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_mixed_forward_reverse_mode(source, device, data):
    if device == "cuda":
        from test_gpu import require_gpu

        require_gpu("torch", source)
    f = gb.to_torch(make_source(source))
    x, y, v = (torch.tensor(data[k], device=device) for k in ("x", "y", "v"))

    def loss(x):
        return f(x, y)[0].sum()

    hess = torch.func.hessian(loss)(x)
    torch.testing.assert_close(hess, torch.diag(-torch.sin(x) * y))
    hvp = torch.func.grad(lambda x: torch.func.jvp(loss, (x,), (v,))[1])(x)
    torch.testing.assert_close(hvp, -torch.sin(x) * y * v)


@pytest.mark.parametrize("source", ["numpy", "jax", "tf"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_low_level_forward_ad(source, device, data):
    if device == "cuda":
        from test_gpu import require_gpu

        require_gpu("torch", source)
    f = gb.to_torch(make_source(source))
    x, y, v = (torch.tensor(data[k], device=device) for k in ("x", "y", "v"))
    # Reverse mode can be disabled while forward mode remains active.
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x, v)
        out = f(dual, y)[0]
        primal, tangent = torch.autograd.forward_ad.unpack_dual(out)
    torch.testing.assert_close(primal, torch.sin(x) * y)
    torch.testing.assert_close(tangent, torch.cos(x) * y * v, rtol=1e-5, atol=1e-6)


def test_torch_forward_mode_integer_output():
    f = gb.to_torch(
        gb.from_numpy(
            lambda x: (np.sin(x), np.argsort(x)),
            jvp=lambda x, t: (np.cos(x) * t, np.zeros(x.shape, dtype=np.int64)),
        )
    )
    x = torch.tensor([0.2, 0.4], dtype=torch.float64)
    (out, order), (tangent, order_tangent) = torch.func.jvp(f, (x,), (torch.ones_like(x),))
    torch.testing.assert_close(out, x.sin())
    torch.testing.assert_close(tangent, x.cos())
    torch.testing.assert_close(order, torch.argsort(x))
    torch.testing.assert_close(order_tangent, torch.zeros_like(order))


def test_torch_warp_has_no_forward_mode(data):
    f = gb.to_torch(make_source("warp"))
    x, y = (torch.tensor(data[k]) for k in ("x", "y"))
    with pytest.raises(gb.MaxOrderError, match="forward-mode"):
        torch.func.jvp(lambda x: f(x, y)[0], (x,), (torch.ones_like(x),))


@pytest.mark.parametrize("context", [contextlib.nullcontext, torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("cache_residuals", [None, False, True])
def test_torch_only_builds_residuals_for_backward(context, requires_grad, cache_residuals):
    calls = []

    def evaluate(x):
        calls.append("eval")
        return (x.sin(),)

    def forward(x):
        calls.append("fwd")
        return (x.sin(),), x

    dfn = gb.DiffFunction(evaluate, fwd=forward, bwd=lambda x, c: (x.cos() * c,), framework="numpy")
    f = gb.to_torch(dfn, cache_residuals=cache_residuals)
    x = torch.tensor([0.2, 0.4], requires_grad=requires_grad)
    with context():
        out = f(x)
    needs_backward = requires_grad and context is contextlib.nullcontext
    assert calls == ["fwd" if needs_backward and cache_residuals is not False else "eval"]
    assert out.requires_grad == needs_backward
    torch.testing.assert_close(out, x.sin())
    if needs_backward and cache_residuals is not False:
        out.sum().backward()
        torch.testing.assert_close(x.grad, x.cos())
