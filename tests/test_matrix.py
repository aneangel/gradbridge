"""Every source x every consumer: values and first-order gradients must match the reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
import warp as wp

import gradbridge as gb

from conftest import SOURCES, make_source, ref_values, ref_vjp

RTOL, ATOL = 1e-9, 1e-11


# ----------------------------------------------------------------------------- consumer runners
def run_torch(dfn, x, y, c1, c2, **kw):
    f = gb.to_torch(dfn, **kw)
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    o1, o2 = f(tx, ty)
    ((o1 * torch.tensor(c1)).sum() + (o2 * torch.tensor(c2)).sum()).backward()
    return (o1.detach().numpy(), o2.detach().numpy()), (tx.grad.numpy(), ty.grad.numpy())


def run_jax(dfn, x, y, c1, c2, jit=False):
    f = gb.to_jax(dfn)

    def loss(x, y):
        o1, o2 = f(x, y)
        return jnp.sum(o1 * c1) + jnp.sum(o2 * c2)

    vals = f(jnp.asarray(x), jnp.asarray(y))
    g = jax.grad(loss, argnums=(0, 1))
    if jit:
        g = jax.jit(g)
    gx, gy = g(jnp.asarray(x), jnp.asarray(y))
    return tuple(np.asarray(v) for v in vals), (np.asarray(gx), np.asarray(gy))


def run_tf(dfn, x, y, c1, c2, graph=False):
    f = gb.to_tf(dfn)

    def step(tx, ty):
        with tf.GradientTape() as tape:
            tape.watch([tx, ty])
            o1, o2 = f(tx, ty)
            loss = tf.reduce_sum(o1 * c1) + tf.reduce_sum(o2 * c2)
        gx, gy = tape.gradient(loss, [tx, ty])
        return o1, o2, gx, gy

    if graph:
        spec = tf.TensorSpec([3], tf.float64)
        step = tf.function(step, input_signature=[spec, spec])
    o1, o2, gx, gy = step(tf.constant(x), tf.constant(y))
    return (o1.numpy(), o2.numpy()), (gx.numpy(), gy.numpy())


def run_numpy(dfn, x, y, c1, c2):
    f = gb.to_numpy(dfn)
    vals = f(x, y)
    grads = f.vjp(x, y, cotangents=(c1, c2))
    return vals, grads


@wp.kernel
def _loss_kernel(
    o1: wp.array(dtype=wp.float64),
    o2: wp.array(dtype=wp.float64),
    c1: wp.array(dtype=wp.float64),
    c2: wp.array(dtype=wp.float64),
    loss: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    wp.atomic_add(loss, 0, o1[i] * c1[i])
    if i == 0:
        wp.atomic_add(loss, 0, o2[0] * c2[0])


def run_warp(dfn, x, y, c1, c2):
    f = gb.to_warp(dfn)
    wx = wp.array(x, dtype=wp.float64, requires_grad=True)
    wy = wp.array(y, dtype=wp.float64, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float64, requires_grad=True)
    tape = wp.Tape()
    with tape:
        o1, o2 = f(wx, wy)
        wp.launch(
            _loss_kernel,
            dim=3,
            inputs=[o1, o2, wp.array(c1, dtype=wp.float64), wp.array(c2, dtype=wp.float64)],
            outputs=[loss],
        )
    tape.backward(loss)
    return (o1.numpy(), o2.numpy()), (wx.grad.numpy(), wy.grad.numpy())


CONSUMERS = {
    "torch": run_torch,
    "jax": run_jax,
    "tf": run_tf,
    "numpy": run_numpy,
    "warp": run_warp,
}


def _check(vals, grads, d):
    rv = ref_values(d["x"], d["y"])
    rg = ref_vjp(d["x"], d["y"], d["c1"], d["c2"])
    for got, want in zip(vals, rv):
        np.testing.assert_allclose(np.asarray(got).reshape(np.shape(want)), want, rtol=RTOL, atol=ATOL)
    for got, want in zip(grads, rg):
        np.testing.assert_allclose(np.asarray(got).reshape(np.shape(want)), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("consumer", list(CONSUMERS))
@pytest.mark.parametrize("source", SOURCES)
def test_matrix(source, consumer, data):
    dfn = make_source(source)
    vals, grads = CONSUMERS[consumer](dfn, data["x"], data["y"], data["c1"], data["c2"])
    _check(vals, grads, data)


@pytest.mark.parametrize("source", SOURCES)
def test_jax_jit(source, data):
    vals, grads = run_jax(make_source(source), data["x"], data["y"], data["c1"], data["c2"], jit=True)
    _check(vals, grads, data)


@pytest.mark.parametrize("source", SOURCES)
def test_tf_function(source, data):
    vals, grads = run_tf(make_source(source), data["x"], data["y"], data["c1"], data["c2"], graph=True)
    _check(vals, grads, data)


@pytest.mark.parametrize("source", SOURCES)
def test_torch_cache_residuals(source, data):
    vals, grads = run_torch(make_source(source), data["x"], data["y"], data["c1"], data["c2"], cache_residuals=True)
    _check(vals, grads, data)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("source", SOURCES)
def test_torch_compile_fullgraph(source, backend, data):
    torch._dynamo.reset()
    f = gb.to_torch(make_source(source))
    c1, c2 = torch.tensor(data["c1"]), torch.tensor(data["c2"])

    @torch.compile(fullgraph=True, backend=backend)
    def loss_fn(x, y):
        o1, o2 = f(x, y)
        return (o1 * c1).sum() + (o2 * c2).sum()

    tx = torch.tensor(data["x"], requires_grad=True)
    ty = torch.tensor(data["y"], requires_grad=True)
    loss_fn(tx, ty).backward()
    _check(ref_values(data["x"], data["y"]), (tx.grad.numpy(), ty.grad.numpy()), data)


def test_convert_one_step(data):
    f = gb.convert(lambda x, y: (jnp.sin(x) * y, jnp.sum(x**2).reshape(1)), src="jax", dst="torch")
    tx = torch.tensor(data["x"], requires_grad=True)
    ty = torch.tensor(data["y"], requires_grad=True)
    o1, o2 = f(tx, ty)
    ((o1 * torch.tensor(data["c1"])).sum() + (o2 * torch.tensor(data["c2"])).sum()).backward()
    _check((o1.detach(), o2.detach()), (tx.grad, ty.grad), data)


def test_chain_across_three_frameworks(data):
    """torch -> (JAX function) -> (TF function) -> torch loss, gradients end to end."""
    jax_part = gb.to_torch(gb.from_jax(lambda x: jnp.sin(x)))
    tf_part = gb.to_torch(gb.from_tf(lambda u, y: u * y))
    tx = torch.tensor(data["x"], requires_grad=True)
    ty = torch.tensor(data["y"], requires_grad=True)
    out = tf_part(jax_part(tx), ty)
    (out * torch.tensor(data["c1"])).sum().backward()
    np.testing.assert_allclose(tx.grad.numpy(), data["c1"] * np.cos(data["x"]) * data["y"], rtol=RTOL)
    np.testing.assert_allclose(ty.grad.numpy(), data["c1"] * np.sin(data["x"]), rtol=RTOL)


def test_gradbridge_protocol_hook(data):
    class MyModel:
        """A third-party object joining the protocol via __gradbridge__."""

        def __gradbridge__(self):
            return gb.from_numpy(lambda x: x**3, vjp=lambda x, c: (3 * x**2 * c,))

    f = gb.to_torch(MyModel())
    tx = torch.tensor(data["x"], requires_grad=True)
    f(tx).sum().backward()
    np.testing.assert_allclose(tx.grad.numpy(), 3 * data["x"] ** 2, rtol=RTOL)
