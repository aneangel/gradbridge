"""vmap, second-order derivatives, forward mode, gradcheck/compare, Warp vector types."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
import warp as wp

import gradbridge as gb

from conftest import SECOND_ORDER_SOURCES, SOURCES, make_source, ref_hvp_x, ref_values, ref_vjp

TOL = {"numpy_fd": dict(rtol=1e-3, atol=1e-3)}
DEFAULT_TOL = dict(rtol=1e-7, atol=1e-9)


def tol(source):
    return TOL.get(source, DEFAULT_TOL)


# ============================================================================= vmap
def _ref_batched(xs, y):
    o1 = np.sin(xs) * y
    o2 = np.sum(xs**2, axis=1, keepdims=True)
    return o1, o2


@pytest.mark.parametrize("source", SOURCES)
def test_jax_vmap_with_grad(source, data):
    f = gb.to_jax(make_source(source))
    xs = np.random.default_rng(0).standard_normal((4, 3))
    y = data["y"]

    def batched_loss(xs):
        o1, o2 = jax.vmap(lambda x: f(x, jnp.asarray(y)))(xs)
        return jnp.sum(o1) + jnp.sum(o2)

    o1, o2 = jax.vmap(lambda x: f(x, jnp.asarray(y)))(jnp.asarray(xs))
    r1, r2 = _ref_batched(xs, y)
    np.testing.assert_allclose(o1, r1, **DEFAULT_TOL)
    np.testing.assert_allclose(o2, r2, **DEFAULT_TOL)
    g = jax.jit(jax.grad(batched_loss))(jnp.asarray(xs))
    np.testing.assert_allclose(g, np.cos(xs) * y + 2 * xs, **DEFAULT_TOL)


@pytest.mark.parametrize("source", SOURCES)
def test_torch_func_vmap_and_grad(source, data):
    f = gb.to_torch(make_source(source))
    xs = torch.tensor(np.random.default_rng(0).standard_normal((4, 3)))
    y = torch.tensor(data["y"])
    o1, o2 = torch.func.vmap(lambda x: f(x, y))(xs)
    r1, r2 = _ref_batched(xs.numpy(), data["y"])
    np.testing.assert_allclose(o1.numpy(), r1, **DEFAULT_TOL)
    np.testing.assert_allclose(o2.numpy(), r2, **DEFAULT_TOL)

    # per-example gradients: vmap(grad(...))
    def per_example_loss(x):
        a, b = f(x, y)
        return a.sum() + b.sum()

    g = torch.func.vmap(torch.func.grad(per_example_loss))(xs)
    np.testing.assert_allclose(g.numpy(), np.cos(xs.numpy()) * data["y"] + 2 * xs.numpy(), **DEFAULT_TOL)


@pytest.mark.parametrize("source", SOURCES)
def test_tf_map_fn_with_grad(source, data):
    f = gb.to_tf(make_source(source))
    xs = tf.constant(np.random.default_rng(0).standard_normal((4, 3)))
    y = tf.constant(data["y"])
    with tf.GradientTape() as tape:
        tape.watch(xs)
        o1, o2 = tf.map_fn(lambda x: f(x, y), xs, fn_output_signature=(tf.float64, tf.float64))
        loss = tf.reduce_sum(o1) + tf.reduce_sum(o2)
    g = tape.gradient(loss, xs)
    r1, r2 = _ref_batched(xs.numpy(), data["y"])
    np.testing.assert_allclose(o1.numpy(), r1, **DEFAULT_TOL)
    np.testing.assert_allclose(o2.numpy(), r2, **DEFAULT_TOL)
    np.testing.assert_allclose(g.numpy(), np.cos(xs.numpy()) * data["y"] + 2 * xs.numpy(), **DEFAULT_TOL)


def test_tf_vectorized_map_forward(data):
    """Forward values work; TF itself cannot backprop py_function inside pfor's loop fallback."""
    f = gb.to_tf(make_source("jax"))
    xs = tf.constant(np.random.default_rng(0).standard_normal((4, 3)))
    o1, o2 = tf.vectorized_map(lambda x: f(x, tf.constant(data["y"])), xs, warn=False)
    r1, r2 = _ref_batched(xs.numpy(), data["y"])
    np.testing.assert_allclose(o1.numpy(), r1, **DEFAULT_TOL)
    np.testing.assert_allclose(o2.numpy(), r2, **DEFAULT_TOL)


@pytest.mark.parametrize("consumer", ["jax", "torch"])
def test_natively_batched_source(consumer):
    """batched=True: one call with a batch dimension instead of a Python loop."""
    calls = []

    def sin_np(x):
        calls.append(x.shape)
        return np.sin(x)

    dfn = gb.from_numpy(sin_np, vjp=lambda x, c: (np.cos(x) * c,), batched=True)
    xs = np.random.default_rng(0).standard_normal((5, 3))
    if consumer == "jax":
        out = jax.vmap(gb.to_jax(dfn))(jnp.asarray(xs))
    else:
        out = torch.func.vmap(gb.to_torch(dfn))(torch.tensor(xs))
    np.testing.assert_allclose(np.asarray(out), np.sin(xs), **DEFAULT_TOL)
    assert (5, 3) in calls  # the whole batch went through in one call


# ============================================================================= second order
@pytest.mark.parametrize("source", SECOND_ORDER_SOURCES + ["numpy"])
def test_torch_double_backward(source, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = gb.to_torch(make_source(source))
        x = torch.tensor(data["x"], requires_grad=True)
        y = torch.tensor(data["y"])
        c1, c2, v = (torch.tensor(data[k]) for k in ("c1", "c2", "v"))
        o1, o2 = f(x, y)
        s = (o1 * c1).sum() + (o2 * c2).sum()
        (g,) = torch.autograd.grad(s, x, create_graph=True)
        np.testing.assert_allclose(g.detach().numpy(), ref_vjp(data["x"], data["y"], data["c1"], data["c2"])[0], **tol(source))
        (hv,) = torch.autograd.grad((g * v).sum(), x)
    np.testing.assert_allclose(hv.numpy(), ref_hvp_x(data["x"], data["y"], data["c1"], data["c2"], data["v"]), **tol(source))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("source", SECOND_ORDER_SOURCES + ["numpy"])
def test_jax_grad_of_grad(source, jit, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = gb.to_jax(make_source(source))
        y = jnp.asarray(data["y"])

        def s(x):
            o1, o2 = f(x, y)
            return jnp.sum(o1 * data["c1"]) + jnp.sum(o2 * data["c2"])

        hvp = jax.grad(lambda x: jnp.vdot(jax.grad(s)(x), data["v"]))
        if jit:
            hvp = jax.jit(hvp)
        hv = hvp(jnp.asarray(data["x"]))
    np.testing.assert_allclose(hv, ref_hvp_x(data["x"], data["y"], data["c1"], data["c2"], data["v"]), **tol(source))


@pytest.mark.parametrize("source", SECOND_ORDER_SOURCES + ["numpy"])
def test_tf_nested_tapes(source, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = gb.to_tf(make_source(source))
        x = tf.constant(data["x"])
        y = tf.constant(data["y"])
        with tf.GradientTape() as outer:
            outer.watch(x)
            with tf.GradientTape() as inner:
                inner.watch(x)
                o1, o2 = f(x, y)
                s = tf.reduce_sum(o1 * data["c1"]) + tf.reduce_sum(o2 * data["c2"])
            g = inner.gradient(s, x)
            gv = tf.reduce_sum(g * data["v"])
        hv = outer.gradient(gv, x)
    np.testing.assert_allclose(hv.numpy(), ref_hvp_x(data["x"], data["y"], data["c1"], data["c2"], data["v"]), **tol(source))


def test_warp_second_order_raises_clearly(data):
    f = gb.to_torch(make_source("warp"))
    x = torch.tensor(data["x"], requires_grad=True)
    o1, o2 = f(x, torch.tensor(data["y"]))
    (g,) = torch.autograd.grad(o1.sum() + o2.sum(), x, create_graph=True)  # first order: fine
    np.testing.assert_allclose(g.detach().numpy(), np.cos(data["x"]) * data["y"] + 2 * data["x"], **DEFAULT_TOL)
    with pytest.raises(Exception, match="first-order"):
        torch.autograd.grad(g.sum(), x)


def test_warp_second_order_raises_in_jax(data):
    f = gb.to_jax(make_source("warp"))
    y = jnp.asarray(data["y"])
    s = lambda x: jnp.sum(f(x, y)[0])
    with pytest.raises(gb.MaxOrderError, match="first-order"):
        jax.grad(lambda x: jnp.sum(jax.grad(s)(x)))(jnp.asarray(data["x"]))


# ============================================================================= forward mode
@pytest.mark.parametrize("source", ["numpy", "torch", "jax", "tf"])
def test_jax_forward_mode(source, data):
    f = gb.to_jax(make_source(source), mode="jvp")
    x, y = jnp.asarray(data["x"]), jnp.asarray(data["y"])
    tx, ty = jnp.asarray(data["v"]), jnp.asarray(data["c1"])
    (o1, o2), (t1, t2) = jax.jvp(f, (x, y), (tx, ty))
    np.testing.assert_allclose(t1, np.cos(data["x"]) * data["y"] * data["v"] + np.sin(data["x"]) * data["c1"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(t2, [np.sum(2 * data["x"] * data["v"])], rtol=1e-6, atol=1e-8)
    # jacfwd under jit
    J = jax.jit(jax.jacfwd(lambda x: f(x, y)[0]))(x)
    np.testing.assert_allclose(J, np.diag(np.cos(data["x"]) * data["y"]), rtol=1e-6, atol=1e-8)


def test_warp_has_no_forward_mode(data):
    f = gb.to_jax(make_source("warp"), mode="jvp")
    with pytest.raises(gb.MaxOrderError, match="forward-mode"):
        jax.jvp(lambda x: f(x, jnp.asarray(data["y"]))[0], (jnp.asarray(data["x"]),), (jnp.ones(3),))


# ============================================================================= gradcheck / compare
@pytest.mark.parametrize("source", SOURCES)
def test_gradcheck_first_order(source, data):
    report = gb.gradcheck(make_source(source), data["x"], data["y"])
    assert report.passed, str(report)


@pytest.mark.parametrize("source", ["torch", "jax", "tf", "numpy"])
def test_gradcheck_second_order(source, data):
    report = gb.gradcheck(make_source(source), data["x"], data["y"], order=2, rtol=1e-5, atol=1e-7)
    assert report.passed, str(report)


def test_gradcheck_catches_wrong_gradient(data):
    bad = gb.from_numpy(lambda x: x**3, vjp=lambda x, c: (2 * x**2 * c,))  # should be 3x^2
    report = gb.gradcheck(bad, data["x"])
    assert not report.passed
    assert "FAILED" in str(report)


def test_compare_two_frameworks(data):
    assert gb.compare(make_source("torch"), make_source("jax"), data["x"], data["y"])
    assert gb.compare(make_source("tf"), make_source("warp"), data["x"], data["y"])
    wrong = gb.from_numpy(
        lambda x, y: (np.sin(x) * y, np.array([np.sum(x**2)])),
        vjp=lambda x, y, c1, c2: (c1 * np.cos(x) * y, c1 * np.sin(x)),  # forgot the 2*x*c2 term
    )
    assert not gb.compare(make_source("jax"), wrong, data["x"], data["y"])


def test_numpy_without_vjp_warns_and_uses_fd(data):
    f = gb.from_numpy(lambda x: np.sin(x))
    with pytest.warns(UserWarning, match="finite differences"):
        g = gb.to_numpy(f).vjp(data["x"], cotangents=(np.ones(3),))
    np.testing.assert_allclose(g[0], np.cos(data["x"]), rtol=1e-6)


# ============================================================================= Warp vector types
@wp.kernel
def _sqnorm(p: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = wp.dot(p[i], p[i])


def test_warp_vector_inputs_into_torch():
    def sqnorm(points):
        out = wp.zeros(points.shape[0], dtype=wp.float64, requires_grad=True)
        wp.launch(_sqnorm, dim=points.shape[0], inputs=[points], outputs=[out])
        return out

    f = gb.to_torch(gb.from_warp(sqnorm, in_dtypes=[wp.vec3d]))
    p = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    f(p).sum().backward()
    np.testing.assert_allclose(p.grad.numpy(), 2 * p.detach().numpy(), **DEFAULT_TOL)


def test_to_numpy_grad_helper(data):
    f = gb.to_numpy(gb.from_jax(lambda x: jnp.sum(jnp.sin(x))))
    np.testing.assert_allclose(f.grad(data["x"]), np.cos(data["x"]), **DEFAULT_TOL)


# ============================================================================= Warp residual reuse detection
def test_interval_overlap_matches_brute_force():
    """The merged-interval overlap check behind Warp residual caching agrees with a brute-force check."""
    from gradbridge.warp import _overlaps

    def merged(ivs):
        # Same representation _touched_ranges produces: sorted, merged (lo, hi) uint64 arrays.
        ivs = sorted(ivs)
        out = []
        for lo, hi in ivs:
            if out and lo < out[-1][1]:
                out[-1][1] = max(out[-1][1], hi)
            else:
                out.append([lo, hi])
        return np.array([a for a, _ in out], np.uint64), np.array([b for _, b in out], np.uint64)

    rng = np.random.default_rng(0)
    for _ in range(2000):
        a = [(int(lo), int(lo + w)) for lo, w in zip(rng.integers(0, 200, rng.integers(1, 6)), rng.integers(1, 20, 6))]
        b = [(int(lo), int(lo + w)) for lo, w in zip(rng.integers(0, 200, rng.integers(1, 6)), rng.integers(1, 20, 6))]
        brute = any(alo < bhi and blo < ahi for alo, ahi in a for blo, bhi in b)
        assert _overlaps(merged(a), merged(b)) == brute, (a, b)
    assert _overlaps(None, merged([(0, 1)])) and _overlaps(merged([(0, 1)]), None)


def test_touched_ranges_merges_views():
    """Ranges cover every view a tape touched, merge duplicates, and ignore plain values."""
    from gradbridge.warp import _touched_ranges

    @wp.kernel
    def _add(a: wp.array(dtype=wp.float64), s: float, out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        out[i] = a[i] + wp.float64(s)

    base = wp.zeros(16, dtype=wp.float64, device="cpu", requires_grad=True)
    out = wp.zeros(8, dtype=wp.float64, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(_add, dim=8, inputs=[base[0:8], 1.0], outputs=[out], device="cpu")
        wp.launch(_add, dim=8, inputs=[base[8:16], 2.0], outputs=[out], device="cpu")
    lo, hi = _touched_ranges(tape)
    ranges = [(int(a), int(b)) for a, b in zip(lo, hi)]

    def covered(start, stop):
        return any(a <= start and stop <= b for a, b in ranges)

    half = base.capacity // 2
    # Each view's bytes are covered (adjacent halves may stay separate: half-open ranges don't overlap).
    assert covered(base.ptr, base.ptr + half) and covered(base.ptr + half, base.ptr + 2 * half), ranges
    assert covered(out.ptr, out.ptr + out.capacity), ranges
    assert len(ranges) <= 3, f"ranges were not merged: {ranges}"
