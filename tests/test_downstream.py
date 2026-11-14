"""Real downstream libraries crossing framework boundaries; install the downstream extra."""

import jax
import numpy as np
import pytest
import tensorflow as tf
import torch
import warp as wp

import gradbridge as gb


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    return request.param


def _require_device(device, *frameworks):
    if device == "cuda":
        from test_gpu import require_gpu

        require_gpu(*frameworks)


def _assert_inference_skips_residuals(dfn, x):
    forward = dfn._fwd

    def unexpected_forward(*xs):
        pytest.fail("inference built backward state")

    dfn._fwd = unexpected_forward
    try:
        with torch.no_grad():
            out = gb.to_torch(dfn)(x)
        assert not out.requires_grad
        assert out.device == x.device
        return out
    finally:
        dfn._fwd = forward


def test_scipy_spline_jacfwd(device):
    _require_device(device, "torch")
    interpolate = pytest.importorskip("scipy.interpolate")
    knots = np.linspace(0.0, 2.0, 9)
    spline = interpolate.CubicSpline(knots, np.sin(knots))
    dfn = gb.from_numpy(
        lambda x: spline(x),
        vjp=lambda x, c: (spline(x, 1) * c,),
        jvp=lambda x, t: spline(x, 1) * t,
    )
    f = gb.to_torch(dfn)
    x = torch.tensor([0.2, 0.6, 1.4], dtype=torch.float64, device=device, requires_grad=True)
    jac = torch.func.jacfwd(f)(x)
    assert jac.device == x.device
    np.testing.assert_allclose(jac.detach().cpu().numpy(), np.diag(spline(x.detach().cpu().numpy(), 1)))
    f(x).sum().backward()
    np.testing.assert_allclose(x.grad.cpu().numpy(), spline(x.detach().cpu().numpy(), 1))
    np.testing.assert_allclose(_assert_inference_skips_residuals(dfn, x).cpu(), spline(x.detach().cpu().numpy()))


def test_torchdiffeq_sensitivities_from_jax(device):
    _require_device(device, "torch", "jax")
    torchdiffeq = pytest.importorskip("torchdiffeq")

    def solve(y0):
        assert y0.device.type == device
        times = torch.tensor([0.0, 1.0], dtype=y0.dtype, device=y0.device)
        return torchdiffeq.odeint(lambda t, y: -0.4 * y, y0, times, method="rk4", options={"step_size": 0.05})[-1]

    dfn = gb.from_torch(solve, device=device)
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    x, v = (jax.device_put(np.array(values, dtype=np.float64), target) for values in ([0.3, 0.7], [0.2, -0.1]))
    fwd = gb.to_jax(dfn, mode="jvp")
    out, tangent = jax.jvp(fwd, (x,), (v,))
    assert out.devices() == tangent.devices() == {target}
    np.testing.assert_allclose(out, np.exp(-0.4) * x, rtol=1e-8)
    np.testing.assert_allclose(tangent, np.exp(-0.4) * v, rtol=1e-8)
    reverse = jax.jit(jax.grad(lambda x: gb.to_jax(dfn)(x).sum()))(x)
    assert reverse.devices() == {target}
    np.testing.assert_allclose(reverse, np.full(2, np.exp(-0.4)), rtol=1e-8)
    tx = torch.tensor(np.asarray(x).copy(), device=device, requires_grad=True)
    np.testing.assert_allclose(_assert_inference_skips_residuals(dfn, tx).cpu(), out, rtol=1e-8)


def test_diffrax_sensitivities_from_torch(device):
    _require_device(device, "torch", "jax")
    diffrax = pytest.importorskip("diffrax")

    def solve(y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -0.4 * y),
            diffrax.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=0.05,
            y0=y0,
            max_steps=32,
            # Diffrax's default adjoint supports reverse mode only.
            adjoint=diffrax.DirectAdjoint(),
        )
        return solution.ys[-1]

    dfn = gb.from_jax(solve)
    f = gb.to_torch(dfn)
    x = torch.tensor([0.3, 0.7], dtype=torch.float64, device=device, requires_grad=True)
    v = torch.tensor([0.2, -0.1], dtype=torch.float64, device=device)
    out, tangent = torch.func.jvp(f, (x,), (v,))
    assert out.device == tangent.device == x.device
    torch.testing.assert_close(out, np.exp(-0.4) * x)
    torch.testing.assert_close(tangent, np.exp(-0.4) * v)
    jac = torch.func.jacfwd(f)(x)
    torch.testing.assert_close(jac, torch.eye(2, dtype=x.dtype, device=x.device) * np.exp(-0.4))
    f(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, np.exp(-0.4)))
    torch.testing.assert_close(_assert_inference_skips_residuals(dfn, x), out.detach())


def test_keras_sensitivities_from_torch(device):
    _require_device(device, "torch", "tf")
    keras = pytest.importorskip("keras")
    target = "/GPU:0" if device == "cuda" else "/CPU:0"
    with tf.device(target):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(2,), dtype="float64"),
                keras.layers.Dense(4, activation="tanh", dtype="float64"),
                keras.layers.Dense(3, dtype="float64"),
            ]
        )
    rng = np.random.default_rng(42)
    model.set_weights([rng.normal(size=w.shape) * 0.2 for w in model.get_weights()])

    def predict(x):
        with tf.device(target):
            return model(x, training=False)

    dfn = gb.from_tf(predict)
    f = gb.to_torch(dfn)
    x = torch.tensor([[0.3, 0.7]], dtype=torch.float64, device=device, requires_grad=True)
    v = torch.tensor([[0.2, -0.1]], dtype=torch.float64, device=device)
    with tf.device(target):
        native_x, native_v = (tf.constant(t.detach().cpu().numpy()) for t in (x, v))
    with tf.GradientTape() as tape:
        tape.watch(native_x)
        with tf.autodiff.ForwardAccumulator(native_x, native_v) as acc:
            native_out = predict(native_x)
    assert target[1:] in native_out.device
    native_jac = tape.jacobian(native_out, native_x).numpy()
    out, tangent = torch.func.jvp(f, (x,), (v,))
    assert out.device == tangent.device == x.device
    np.testing.assert_allclose(out.detach().cpu().numpy(), native_out.numpy(), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(tangent.detach().cpu().numpy(), acc.jvp(native_out).numpy(), rtol=1e-9, atol=1e-11)
    jac = torch.func.jacfwd(f)(x)
    np.testing.assert_allclose(jac.detach().cpu().numpy(), native_jac, rtol=1e-9, atol=1e-11)
    f(x).sum().backward()
    np.testing.assert_allclose(x.grad.cpu().numpy(), native_jac.sum(axis=(0, 1)), rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(_assert_inference_skips_residuals(dfn, x), out.detach())


@wp.kernel
def _set_velocity(v: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3)):
    qd[0] = v[0]


@wp.kernel
def _position(q: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    out[0] = q[0]


def test_newton_rollout_inference_and_backward(device):
    _require_device(device, "torch", "warp")
    newton = pytest.importorskip("newton")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(), mass=1.0)
        model = builder.finalize(requires_grad=True)
        solver = newton.solvers.SolverSemiImplicit(model)
        steps, dt = 16, 1.0 / 120.0
        states = [model.state() for _ in range(steps + 1)]
        control, contacts = model.control(), model.contacts()

        def rollout(v):
            wp.launch(_set_velocity, dim=1, inputs=[v], outputs=[states[0].particle_qd])
            for i in range(steps):
                states[i].clear_forces()
                solver.step(states[i], states[i + 1], control, contacts, dt)
            out = wp.empty(1, dtype=wp.vec3, requires_grad=True)
            wp.launch(_position, dim=1, inputs=[states[-1].particle_q], outputs=[out])
            return out

        velocity = np.array([[0.3, 0.7, 1.2]], dtype=np.float32)
        native_v = wp.array(velocity, dtype=wp.vec3, requires_grad=True)
        with wp.Tape() as tape:
            native_out = rollout(native_v)
        tape.backward(grads={native_out: wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3)})
        native_grad = native_v.grad.numpy().copy()
        expected_out = native_out.numpy().copy()
        tape.zero()

        dfn = gb.from_warp(rollout, in_dtypes=[wp.vec3], device=device)
        f = gb.to_torch(dfn)
        x = torch.tensor(velocity, device=device, requires_grad=True)
        out = f(x)
        assert out.device == x.device
        np.testing.assert_allclose(out.detach().cpu().numpy(), expected_out, rtol=1e-6)
        out.sum().backward()
        np.testing.assert_allclose(x.grad.cpu().numpy(), native_grad, rtol=1e-6)
        np.testing.assert_allclose(x.grad.cpu().numpy(), np.full_like(velocity, steps * dt), rtol=1e-6)
        np.testing.assert_allclose(_assert_inference_skips_residuals(dfn, x).cpu(), expected_out, rtol=1e-6)
