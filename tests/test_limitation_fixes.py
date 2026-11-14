"""Regression checks for native callbacks and compiled forward-mode derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
import warp as wp

import gradbridge as gb

from conftest import make_source


def _require(device, *frameworks):
    if device == "cuda":
        from test_gpu import require_gpu

        require_gpu(*frameworks)


@pytest.mark.parametrize("consumer", ["jax", "tf"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(), (3,), (2, 1, 2, 1, 3), (0, 3)])
def test_native_torch_callback_preserves_device_and_rank(consumer, device, shape, monkeypatch):
    _require(device, "torch", consumer)
    seen = []

    def source(x):
        if x.device.type != "meta":
            seen.append((x.device.type, tuple(x.shape)))
        return x * x + 2 * x

    def no_host_callback(*args, **kwargs):
        pytest.fail("a supported native callback fell back to a host callback")

    data = np.arange(np.prod(shape, dtype=int), dtype=np.float64).reshape(shape) / 10
    dfn = gb.from_torch(source)
    if consumer == "jax":
        monkeypatch.setattr(jax, "pure_callback", no_host_callback)
        target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
        x = jax.device_put(data, target)
        f = gb.to_jax(dfn)
        out = jax.jit(f)(x)
        grad = jax.jit(jax.grad(lambda x: f(x).sum()))(x)
        assert out.devices() == grad.devices() == {target}
        out, grad = np.asarray(out), np.asarray(grad)
    else:
        monkeypatch.setattr(tf, "py_function", no_host_callback)
        target = "/GPU:0" if device == "cuda" else "/CPU:0"
        f = gb.to_tf(dfn)

        @tf.function(autograph=False)
        def step(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                out = f(x)
                loss = tf.reduce_sum(out)
            return out, tape.gradient(loss, x)

        with tf.device(target):
            out, grad = step(tf.constant(data))
        assert target[1:] in out.device and target[1:] in grad.device
        out, grad = out.numpy(), grad.numpy()
    np.testing.assert_allclose(out, data * data + 2 * data)
    np.testing.assert_allclose(grad, 2 * data + 2)
    assert seen and all(entry == (device, shape) for entry in seen)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_batched_torch_callback_is_one_native_call(device):
    _require(device, "torch", "jax")
    seen = []

    def source(x, y):
        if x.device.type != "meta":
            seen.append((x.device.type, tuple(x.shape)))
        return torch.sin(x) * y

    dfn = gb.from_torch(source, batched=True)
    f = gb.to_jax(dfn)
    target = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    xs = jax.device_put(np.arange(15, dtype=np.float64).reshape(5, 3) / 10, target)
    y = jax.device_put(np.array([0.2, 0.4, 0.6]), target)
    mapped = jax.jit(jax.vmap(f, in_axes=(0, None)))
    out = mapped(xs, y)
    jax.block_until_ready(out)
    assert seen == [(device, (5, 3))]
    np.testing.assert_allclose(out, jnp.sin(xs) * y)
    seen.clear()
    grad = jax.jit(jax.grad(lambda x: mapped(x, y).sum()))(xs)
    jax.block_until_ready(grad)
    np.testing.assert_allclose(grad, jnp.cos(xs) * y)
    assert seen and all(entry == (device, (5, 3)) for entry in seen)


@pytest.mark.parametrize("source", ["numpy", "jax", "tf"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_jvp_and_jacfwd(source, device, backend, data):
    _require(device, "torch", source)
    torch._dynamo.reset()
    f = gb.to_torch(make_source(source))
    x, y, v = (torch.tensor(data[k], device=device) for k in ("x", "y", "v"))

    @torch.compile(fullgraph=True, backend=backend)
    def step(x, v):
        return torch.func.jvp(lambda x: f(x, y), (x,), (v,))

    values, tangents = step(x, v)
    torch.testing.assert_close(values[0], torch.sin(x) * y)
    torch.testing.assert_close(values[1], (x * x).sum().reshape(1))
    torch.testing.assert_close(tangents[0], torch.cos(x) * y * v, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(tangents[1], (2 * x * v).sum().reshape(1), rtol=1e-5, atol=1e-7)
    jacobian = torch.compile(torch.func.jacfwd(lambda x: f(x, y)[0]), fullgraph=True, backend=backend)(x)
    torch.testing.assert_close(jacobian, torch.diag(torch.cos(x) * y), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_external_warp_mutation_invalidates_only_affected_tapes(device, monkeypatch):
    _require(device, "torch", "warp")
    from test_advanced import _square_kernel, _warp_square

    calls = [0, 0]
    buffers = [wp.empty(3, dtype=wp.float64, device=device, requires_grad=True) for _ in range(2)]

    def make(index):
        def source(x):
            calls[index] += 1
            wp.launch(_square_kernel, dim=3, inputs=[x], outputs=[buffers[index]], device=x.device)
            return _warp_square(buffers[index])

        return gb.to_torch(gb.from_warp(source, device=device))

    a, b = make(0), make(1)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64, device=device, requires_grad=True)
    oa, ob = a(x), b(y)
    assert calls == [1, 1]
    buffers[0].fill_(99.0)

    def no_sync(*args, **kwargs):
        pytest.fail("mutation notification synchronized the GPU")

    with monkeypatch.context() as patch:
        patch.setattr(wp, "synchronize_device", no_sync)
        patch.setattr(wp, "synchronize_stream", no_sync)
        gb.invalidate_warp_residuals(buffers[0])
    (oa.sum() + ob.sum()).backward()
    assert calls == [2, 1]
    torch.testing.assert_close(x.grad, 4 * x.detach() ** 3, rtol=0, atol=0)
    torch.testing.assert_close(y.grad, 4 * y.detach() ** 3, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("source", ["torch", "jax", "warp"])
@pytest.mark.parametrize("graph", [False, True])
def test_tf_without_compiler_preserves_buffers(source, device, graph, data, monkeypatch):
    _require(device, "tf", source)
    from gradbridge import _convert, _tf_native

    monkeypatch.setattr(_tf_native, "load", lambda: None)
    f = gb.to_tf(make_source(source))

    def step(x, y):
        with tf.GradientTape() as tape:
            tape.watch([x, y])
            a, b = f(x, y)
            loss = tf.reduce_sum(a) + tf.reduce_sum(b)
        return a, b, *tape.gradient(loss, [x, y])

    if graph:
        step = tf.function(step, autograph=False)
    target = "/GPU:0" if device == "cuda" else "/CPU:0"
    with wp.ScopedDevice(device), tf.device(target):
        x, y = (tf.constant(data[k]) for k in ("x", "y"))
        step(x, y)  # cache shape inference before checking the steady-state path

        def no_numpy_copy(*args, **kwargs):
            pytest.fail("the compiler-free callback copied through NumPy")

        with monkeypatch.context() as patch:
            if device == "cuda":
                patch.setattr(_convert, "to_numpy", no_numpy_copy)
            a, b, gx, gy = step(x, y)
        assert all(target[1:] in t.device for t in (a, b, gx, gy))
    np.testing.assert_allclose(a.numpy(), np.sin(data["x"]) * data["y"], rtol=1e-9)
    np.testing.assert_allclose(b.numpy(), [np.sum(data["x"] ** 2)], rtol=1e-9)
    np.testing.assert_allclose(gx.numpy(), np.cos(data["x"]) * data["y"] + 2 * data["x"], rtol=1e-9)
    np.testing.assert_allclose(gy.numpy(), np.sin(data["x"]), rtol=1e-9)


def test_tf_python_callback_shares_gpu_pointers(monkeypatch):
    _require("cuda", "tf", "torch")
    from gradbridge import _convert, _tf_native

    seen = []

    fallback = _tf_native.NativeBridge(
        "test TensorFlow fallback", _tf_native._TF_DTYPES, fallback_warning=RuntimeWarning
    )

    def unavailable():
        raise FileNotFoundError("test compiler is unavailable")

    def source(x):
        out = x * x
        if x.device.type != "meta":
            seen.append((x.data_ptr(), out.data_ptr()))
        return out

    monkeypatch.setattr(_tf_native, "load", lambda: fallback.load(unavailable))
    f = gb.to_tf(gb.from_torch(source))
    with tf.device("/GPU:0"):
        x = tf.constant([0.2, 0.4], tf.float64)
        original = _convert.to_torch(x)
        with pytest.warns(RuntimeWarning, match="could not build"):
            out = f(x)
    returned = _convert.to_torch(out)
    assert seen == [(original.data_ptr(), returned.data_ptr())]
    torch.testing.assert_close(returned, original * original)


@pytest.mark.parametrize("consumer", ["tf", "warp"])
@pytest.mark.parametrize("source", ["torch", "jax", "tf", "numpy"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_jvp_rule_in_tf_and_warp(consumer, source, device, data):
    _require(device, consumer, source)
    rule = make_source(source).jvp_function
    target = "/GPU:0" if device == "cuda" else "/CPU:0"
    with wp.ScopedDevice(device), tf.device(target):
        if consumer == "tf":
            inputs = [tf.constant(data[k]) for k in ("x", "y", "v", "c1")]
            tangents = gb.to_tf(rule)(*inputs)
            assert all(target[1:] in t.device for t in tangents)
        else:
            inputs = [wp.array(data[k], dtype=wp.float64, device=device) for k in ("x", "y", "v", "c1")]
            tangents = gb.to_warp(rule, device=device)(*inputs)
            assert all(t.device == wp.get_device(device) for t in tangents)
    np.testing.assert_allclose(
        tangents[0].numpy(),
        np.cos(data["x"]) * data["y"] * data["v"] + np.sin(data["x"]) * data["c1"],
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(tangents[1].numpy(), [np.sum(2 * data["x"] * data["v"])], rtol=1e-5, atol=1e-7)


def test_compiled_nested_jvp_rejects_instead_of_returning_zero():
    torch._dynamo.reset()
    f = gb.to_torch(gb.from_jax(lambda x: jnp.sin(x)))
    x = torch.tensor([0.2, 0.4])
    v = torch.ones_like(x)

    def first(x):
        return torch.func.jvp(f, (x,), (v,))[1]

    compiled = torch.compile(lambda x: torch.func.jvp(first, (x,), (v,))[1], fullgraph=True, backend="aot_eager")
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled(x)


@wp.kernel
def _array_control(x: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = x[i] * scale[0]


def test_cuda_graph_control_values_can_change_as_arrays(monkeypatch):
    _require("cuda", "torch", "warp")
    launches = []
    launch = wp.capture_launch

    def counted(*args, **kwargs):
        launches.append(1)
        return launch(*args, **kwargs)

    def source(x, scale):
        out = wp.empty_like(x, requires_grad=True)
        wp.launch(_array_control, dim=x.shape[0], inputs=[x, scale], outputs=[out], device=x.device)
        return out

    monkeypatch.setattr(wp, "capture_launch", counted)
    f = gb.to_torch(gb.from_warp(source, capture=True))
    with wp.ScopedDevice("cuda"):
        for scale in (0.5, 0.5, 0.75, 0.25):
            x = torch.arange(1.0, 4.0, dtype=torch.float64, device="cuda", requires_grad=True)
            out = f(x, torch.tensor([scale], dtype=x.dtype, device=x.device))
            out.sum().backward()
            torch.testing.assert_close(out, x.detach() * scale, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, torch.full_like(x, scale), rtol=0, atol=0)
    assert len(launches) >= 3


def test_warp_conversion_honors_explicit_device():
    _require("cuda", "warp")
    from gradbridge import _convert

    cpu = wp.array([1.0, 2.0, 3.0], device="cpu")
    assert _convert.to_warp(cpu) is cpu
    gpu = _convert.to_warp(cpu, device="cuda:0")
    assert gpu.device == wp.get_device("cuda:0")
    assert _convert.to_warp(gpu, device="cuda:0") is gpu
    back = _convert.to_warp(gpu, device="cpu")
    assert back.device == wp.get_device("cpu")
    np.testing.assert_array_equal(back.numpy(), cpu.numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("jit", [False, True])
def test_jax_native_tensorflow_graph_supports_xla(device, jit, data, monkeypatch):
    _require(device, "tf", "jax")
    f = gb.to_tf(make_source("jax"), native=True)

    def no_callback(*args, **kwargs):
        pytest.fail("native graph lowering executed a gradbridge callback")

    from gradbridge import _tf_native

    monkeypatch.setattr(_tf_native, "load", no_callback)
    monkeypatch.setattr(tf, "py_function", no_callback)

    def step(x, y):
        with tf.GradientTape() as outer:
            outer.watch(x)
            with tf.GradientTape() as inner:
                inner.watch(x)
                a, b = f(x, y)
                loss = tf.reduce_sum(a) + tf.reduce_sum(b)
            grad = inner.gradient(loss, x)
        return a, b, grad, outer.gradient(grad, x)

    call = tf.function(step, jit_compile=True, autograph=False) if jit else step
    target = "/GPU:0" if device == "cuda" else "/CPU:0"
    with tf.device(target):
        a, b, grad, hess = call(tf.constant(data["x"]), tf.constant(data["y"]))
    assert all(target[1:] in t.device for t in (a, b, grad, hess))
    np.testing.assert_allclose(a.numpy(), np.sin(data["x"]) * data["y"], rtol=1e-9)
    np.testing.assert_allclose(b.numpy(), [np.sum(data["x"] ** 2)], rtol=1e-9)
    np.testing.assert_allclose(grad.numpy(), np.cos(data["x"]) * data["y"] + 2 * data["x"], rtol=1e-9)
    np.testing.assert_allclose(hess.numpy(), -np.sin(data["x"]) * data["y"] + 2, rtol=1e-9)


def test_native_tensorflow_graph_rejects_unexportable_source():
    with pytest.raises(ValueError, match="JAX or TensorFlow source"):
        gb.to_tf(gb.from_numpy(lambda x: x * x), native=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_tensorflow_callback_does_not_require_warp(device, monkeypatch):
    _require(device, "tf", "jax")
    import gradbridge.tf as gbtf
    from gradbridge import _tf_native

    def no_native_handler():
        pytest.fail("a JAX/TensorFlow-only bridge tried to use the Warp-based handler")

    monkeypatch.setattr(gbtf, "_HAS_WARP", False)
    monkeypatch.setattr(_tf_native, "load", no_native_handler)
    f = gb.to_tf(gb.from_jax(lambda x: jnp.sin(x)))
    target = "/GPU:0" if device == "cuda" else "/CPU:0"
    with tf.device(target):
        x = tf.constant([0.2, 0.4], tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = f(x)
        grad = tape.gradient(out, x)
    np.testing.assert_allclose(out.numpy(), np.sin([0.2, 0.4]), rtol=1e-9)
    np.testing.assert_allclose(grad.numpy(), np.cos([0.2, 0.4]), rtol=1e-9)
    assert target[1:] in out.device and target[1:] in grad.device
