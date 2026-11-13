"""GPU versions of test_advanced: vmap, second-order, forward mode, max-order errors.

Inputs live on the GPU; results must stay there and keep the input dtype (float32 and float64).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

import gradbridge as gb

from conftest import ref_hvp_x, ref_vjp
from test_gpu import DTYPES, GPU_SOURCES, TOLS, gpu_device, gpu_source, require_gpu

SECOND_ORDER = ["torch", "jax", "tf", "numpy"]
# The numpy source has no second-order rule, so its Hessian comes from finite differences.
FD_TOLS = {"float32": dict(rtol=5e-2, atol=5e-2), "float64": dict(rtol=1e-3, atol=1e-3)}


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.fixture
def gdata(data, dtype):
    return {k: v.astype(dtype) for k, v in data.items()}


def _tol(source, dtype):
    return FD_TOLS[dtype] if source == "numpy" else TOLS[dtype]


def _assert_on_gpu(*arrays, dtype):
    gpu = gpu_device()
    for a in arrays:
        if isinstance(a, torch.Tensor):
            assert a.device.type == "cuda", f"torch result on {a.device}"
        elif isinstance(a, jax.Array):
            assert a.devices() == {gpu}, f"JAX result on {a.devices()}"
        else:
            assert "GPU" in a.device, f"TF result on {a.device}"
        assert np.dtype(str(a.dtype).replace("torch.", "").replace("<dtype: '", "").replace("'>", "")) == dtype


def _ref_batched(xs, y):
    return np.sin(xs) * y, np.sum(xs**2, axis=1, keepdims=True)


def _np64(a):
    return np.asarray(gb._convert.to_numpy(a), dtype=np.float64)


# ============================================================================= vmap
@pytest.mark.parametrize("source", GPU_SOURCES)
def test_jax_vmap_with_grad_gpu(source, gdata, dtype):
    require_gpu("jax", source)
    f = gb.to_jax(gpu_source(source))
    gpu = gpu_device()
    xs_np = np.random.default_rng(0).standard_normal((4, 3)).astype(dtype)
    xs, y = jax.device_put(xs_np, gpu), jax.device_put(gdata["y"], gpu)

    def batched_loss(xs):
        o1, o2 = jax.vmap(lambda x: f(x, y))(xs)
        return jnp.sum(o1) + jnp.sum(o2)

    o1, o2 = jax.vmap(lambda x: f(x, y))(xs)
    g = jax.jit(jax.grad(batched_loss))(xs)
    _assert_on_gpu(o1, o2, g, dtype=dtype)
    y64, xs64 = gdata["y"].astype(np.float64), xs_np.astype(np.float64)
    r1, r2 = _ref_batched(xs64, y64)
    np.testing.assert_allclose(_np64(o1), r1, **TOLS[dtype])
    np.testing.assert_allclose(_np64(o2), r2, **TOLS[dtype])
    np.testing.assert_allclose(_np64(g), np.cos(xs64) * y64 + 2 * xs64, **TOLS[dtype])


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_torch_func_vmap_and_grad_gpu(source, gdata, dtype):
    require_gpu("torch", source)
    f = gb.to_torch(gpu_source(source))
    xs_np = np.random.default_rng(0).standard_normal((4, 3)).astype(dtype)
    xs, y = torch.tensor(xs_np, device="cuda"), torch.tensor(gdata["y"], device="cuda")
    o1, o2 = torch.func.vmap(lambda x: f(x, y))(xs)

    def per_example_loss(x):
        a, b = f(x, y)
        return a.sum() + b.sum()

    g = torch.func.vmap(torch.func.grad(per_example_loss))(xs)
    _assert_on_gpu(o1, o2, g, dtype=dtype)
    y64, xs64 = gdata["y"].astype(np.float64), xs_np.astype(np.float64)
    r1, r2 = _ref_batched(xs64, y64)
    np.testing.assert_allclose(_np64(o1), r1, **TOLS[dtype])
    np.testing.assert_allclose(_np64(o2), r2, **TOLS[dtype])
    np.testing.assert_allclose(_np64(g), np.cos(xs64) * y64 + 2 * xs64, **TOLS[dtype])


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_tf_map_fn_with_grad_gpu(source, gdata, dtype):
    require_gpu("tf", source)
    f = gb.to_tf(gpu_source(source))
    xs_np = np.random.default_rng(0).standard_normal((4, 3)).astype(dtype)
    tdt = tf.as_dtype(dtype)
    with tf.device("/GPU:0"):
        xs, y = tf.constant(xs_np), tf.constant(gdata["y"])
        with tf.GradientTape() as tape:
            tape.watch(xs)
            o1, o2 = tf.map_fn(lambda x: f(x, y), xs, fn_output_signature=(tdt, tdt))
            loss = tf.reduce_sum(o1) + tf.reduce_sum(o2)
        g = tape.gradient(loss, xs)
    _assert_on_gpu(o1, o2, g, dtype=dtype)
    y64, xs64 = gdata["y"].astype(np.float64), xs_np.astype(np.float64)
    r1, r2 = _ref_batched(xs64, y64)
    np.testing.assert_allclose(_np64(o1), r1, **TOLS[dtype])
    np.testing.assert_allclose(_np64(o2), r2, **TOLS[dtype])
    np.testing.assert_allclose(_np64(g), np.cos(xs64) * y64 + 2 * xs64, **TOLS[dtype])


@pytest.mark.parametrize("consumer", ["jax", "torch"])
def test_natively_batched_source_gpu(consumer, dtype):
    """batched=True: the whole batch reaches the source in one call, on GPU inputs."""
    require_gpu(consumer)
    calls = []

    def sin_np(x):
        calls.append(x.shape)
        return np.sin(x)

    dfn = gb.from_numpy(sin_np, vjp=lambda x, c: (np.cos(x) * c,), batched=True)
    xs_np = np.random.default_rng(0).standard_normal((5, 3)).astype(dtype)
    if consumer == "jax":
        out = jax.vmap(gb.to_jax(dfn))(jax.device_put(xs_np, gpu_device()))
    else:
        out = torch.func.vmap(gb.to_torch(dfn))(torch.tensor(xs_np, device="cuda"))
    _assert_on_gpu(out, dtype=dtype)
    np.testing.assert_allclose(_np64(out), np.sin(xs_np.astype(np.float64)), **TOLS[dtype])
    assert (5, 3) in calls


# ============================================================================= second order
def _ref_hvp(d):
    d64 = {k: v.astype(np.float64) for k, v in d.items()}
    return ref_hvp_x(d64["x"], d64["y"], d64["c1"], d64["c2"], d64["v"]), d64


@pytest.mark.parametrize("source", SECOND_ORDER)
def test_torch_double_backward_gpu(source, gdata, dtype):
    require_gpu("torch", source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = gb.to_torch(gpu_source(source))
        x = torch.tensor(gdata["x"], device="cuda", requires_grad=True)
        y, c1, c2, v = (torch.tensor(gdata[k], device="cuda") for k in ("y", "c1", "c2", "v"))
        o1, o2 = f(x, y)
        (g,) = torch.autograd.grad((o1 * c1).sum() + (o2 * c2).sum(), x, create_graph=True)
        (hv,) = torch.autograd.grad((g * v).sum(), x)
    _assert_on_gpu(g, hv, dtype=dtype)
    ref, d64 = _ref_hvp(gdata)
    np.testing.assert_allclose(_np64(g), ref_vjp(d64["x"], d64["y"], d64["c1"], d64["c2"])[0], **_tol(source, dtype))
    np.testing.assert_allclose(_np64(hv), ref, **_tol(source, dtype))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("source", SECOND_ORDER)
def test_jax_grad_of_grad_gpu(source, jit, gdata, dtype):
    require_gpu("jax", source)
    gpu = gpu_device()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = gb.to_jax(gpu_source(source))
        y, c1, c2, v = (jax.device_put(gdata[k], gpu) for k in ("y", "c1", "c2", "v"))

        def s(x):
            o1, o2 = f(x, y)
            return jnp.sum(o1 * c1) + jnp.sum(o2 * c2)

        hvp = jax.grad(lambda x: jnp.vdot(jax.grad(s)(x), v))
        if jit:
            hvp = jax.jit(hvp)
        hv = hvp(jax.device_put(gdata["x"], gpu))
    _assert_on_gpu(hv, dtype=dtype)
    np.testing.assert_allclose(_np64(hv), _ref_hvp(gdata)[0], **_tol(source, dtype))


@pytest.mark.parametrize("source", SECOND_ORDER)
def test_tf_nested_tapes_gpu(source, gdata, dtype):
    require_gpu("tf", source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = gb.to_tf(gpu_source(source))
        with tf.device("/GPU:0"):
            x, y = tf.constant(gdata["x"]), tf.constant(gdata["y"])
            with tf.GradientTape() as outer:
                outer.watch(x)
                with tf.GradientTape() as inner:
                    inner.watch(x)
                    o1, o2 = f(x, y)
                    s = tf.reduce_sum(o1 * gdata["c1"]) + tf.reduce_sum(o2 * gdata["c2"])
                g = inner.gradient(s, x)
                gv = tf.reduce_sum(g * gdata["v"])
            hv = outer.gradient(gv, x)
    _assert_on_gpu(g, hv, dtype=dtype)
    np.testing.assert_allclose(_np64(hv), _ref_hvp(gdata)[0], **_tol(source, dtype))


def test_warp_second_order_raises_clearly_gpu(gdata, dtype):
    require_gpu("torch", "warp")
    f = gb.to_torch(gpu_source("warp"))
    x = torch.tensor(gdata["x"], device="cuda", requires_grad=True)
    o1, o2 = f(x, torch.tensor(gdata["y"], device="cuda"))
    (g,) = torch.autograd.grad(o1.sum() + o2.sum(), x, create_graph=True)
    _assert_on_gpu(g, dtype=dtype)
    with pytest.raises(Exception, match="first-order"):
        torch.autograd.grad(g.sum(), x)


def test_warp_second_order_raises_in_jax_gpu(gdata):
    require_gpu("jax", "warp")
    f = gb.to_jax(gpu_source("warp"))
    gpu = gpu_device()
    y = jax.device_put(gdata["y"], gpu)
    s = lambda x: jnp.sum(f(x, y)[0])
    with pytest.raises(gb.MaxOrderError, match="first-order"):
        jax.grad(lambda x: jnp.sum(jax.grad(s)(x)))(jax.device_put(gdata["x"], gpu))


# ============================================================================= forward mode
@pytest.mark.parametrize("source", ["numpy", "torch", "jax", "tf"])
def test_jax_forward_mode_gpu(source, gdata, dtype):
    require_gpu("jax", source)
    gpu = gpu_device()
    f = gb.to_jax(gpu_source(source), mode="jvp")
    x, y, tx, ty = (jax.device_put(gdata[k], gpu) for k in ("x", "y", "v", "c1"))
    (o1, o2), (t1, t2) = jax.jvp(f, (x, y), (tx, ty))
    J = jax.jit(jax.jacfwd(lambda x: f(x, y)[0]))(x)
    _assert_on_gpu(o1, o2, t1, t2, J, dtype=dtype)
    d = {k: v.astype(np.float64) for k, v in gdata.items()}
    np.testing.assert_allclose(_np64(t1), np.cos(d["x"]) * d["y"] * d["v"] + np.sin(d["x"]) * d["c1"], **TOLS[dtype])
    np.testing.assert_allclose(_np64(t2), [np.sum(2 * d["x"] * d["v"])], **TOLS[dtype])
    np.testing.assert_allclose(_np64(J), np.diag(np.cos(d["x"]) * d["y"]), **TOLS[dtype])


def test_warp_has_no_forward_mode_gpu(gdata):
    require_gpu("jax", "warp")
    gpu = gpu_device()
    f = gb.to_jax(gpu_source("warp"), mode="jvp")
    x, y = jax.device_put(gdata["x"], gpu), jax.device_put(gdata["y"], gpu)
    with pytest.raises(gb.MaxOrderError, match="forward-mode"):
        jax.jvp(lambda x: f(x, y)[0], (x,), (jnp.ones_like(x),))
