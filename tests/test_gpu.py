"""GPU coverage: inputs live on the device and outputs/gradients must come back on it, in the input dtype.

Every test runs in float32 and float64. Each test calls ``require_gpu`` with the frameworks it
touches and is skipped only when one of those has no GPU, so e.g. a CPU-only TensorFlow build
skips the TensorFlow tests and nothing else.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
import warp as wp

import gradbridge as gb

from conftest import make_source, ref_values, ref_vjp

HAS_GPU = {
    "torch": torch.cuda.is_available(),
    "jax": any(d.platform == "gpu" for d in jax.devices()),
    "tf": bool(tf.config.list_physical_devices("GPU")),
    "warp": wp.get_cuda_device_count() > 0,
    "numpy": True,  # host-only by design; its inputs and outputs are converted by the consumer
}


def require_gpu(*frameworks):
    """Skip unless every framework this test touches can run on the GPU."""
    missing = sorted({fw for fw in frameworks if not HAS_GPU[fw]})
    if missing:
        pytest.skip(f"no GPU for {', '.join(missing)}")

GPU_SOURCES = ["numpy", "torch", "jax", "tf", "warp"]
DTYPES = ["float32", "float64"]
# float32 tolerances sit a few ulps above machine epsilon (1.2e-7); float64 matches test_matrix.
TOLS = {"float32": dict(rtol=1e-5, atol=1e-6), "float64": dict(rtol=1e-9, atol=1e-11)}


def gpu_device():
    return [d for d in jax.devices() if d.platform == "gpu"][0]


# ----------------------------------------------------------------------------- Warp source per dtype
@wp.kernel
def _warp_kernel32(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    out1: wp.array(dtype=wp.float32),
    out2: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out1[i] = wp.sin(x[i]) * y[i]
    wp.atomic_add(out2, 0, x[i] * x[i])


@wp.kernel
def _warp_kernel64(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    out1: wp.array(dtype=wp.float64),
    out2: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    out1[i] = wp.sin(x[i]) * y[i]
    wp.atomic_add(out2, 0, x[i] * x[i])


def _warp_fn(x, y):
    kernel = _warp_kernel32 if x.dtype == wp.float32 else _warp_kernel64
    out1 = wp.zeros(x.shape, dtype=x.dtype, requires_grad=True, device=x.device)
    out2 = wp.zeros(1, dtype=x.dtype, requires_grad=True, device=x.device)
    wp.launch(kernel, dim=x.shape[0], inputs=[x, y], outputs=[out1, out2], device=x.device)
    return out1, out2


def gpu_source(name):
    """Like ``conftest.make_source`` but the Warp source dispatches on the input dtype."""
    if name == "warp":
        return gb.from_warp(_warp_fn, name="F_warp_any")
    return make_source(name)


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.fixture
def gdata(data, dtype):
    return {k: v.astype(dtype) for k, v in data.items()}


# ----------------------------------------------------------------------------- checks
def _np(a):
    return np.asarray(gb._convert.to_numpy(a))


def _check(vals, grads, d, dtype):
    """Values and gradients match the float64 reference and keep the input dtype."""
    d64 = {k: v.astype(np.float64) for k, v in d.items()}
    for got, want in zip(vals, ref_values(d64["x"], d64["y"])):
        got = _np(got)
        assert got.dtype == np.dtype(dtype), f"output dtype {got.dtype} != {dtype}"
        np.testing.assert_allclose(got.reshape(np.shape(want)), want, **TOLS[dtype])
    for got, want in zip(grads, ref_vjp(d64["x"], d64["y"], d64["c1"], d64["c2"])):
        got = _np(got)
        assert got.dtype == np.dtype(dtype), f"gradient dtype {got.dtype} != {dtype}"
        np.testing.assert_allclose(got.reshape(np.shape(want)), want, **TOLS[dtype])


# ----------------------------------------------------------------------------- consumer runners
def _run_torch_cuda(dfn, d, **kw):
    f = gb.to_torch(dfn, **kw)
    tx = torch.tensor(d["x"], device="cuda", requires_grad=True)
    ty = torch.tensor(d["y"], device="cuda", requires_grad=True)
    o1, o2 = f(tx, ty)
    assert o1.device.type == "cuda" and o2.device.type == "cuda", "outputs left the GPU"
    c1, c2 = torch.tensor(d["c1"], device="cuda"), torch.tensor(d["c2"], device="cuda")
    ((o1 * c1).sum() + (o2 * c2).sum()).backward()
    assert tx.grad.device.type == "cuda" and ty.grad.device.type == "cuda", "gradients left the GPU"
    return (o1.detach(), o2.detach()), (tx.grad, ty.grad)


def _run_jax_gpu(dfn, d, jit=False):
    f = gb.to_jax(dfn)
    gpu = gpu_device()
    x, y = jax.device_put(d["x"], gpu), jax.device_put(d["y"], gpu)
    c1, c2 = jnp.asarray(d["c1"]), jnp.asarray(d["c2"])

    def loss(x, y):
        o1, o2 = f(x, y)
        return jnp.sum(o1 * c1) + jnp.sum(o2 * c2)

    vals = f(x, y)
    g = jax.grad(loss, argnums=(0, 1))
    if jit:
        g = jax.jit(g)
    gx, gy = g(x, y)
    assert all(v.devices() == {gpu} for v in vals), "outputs left the GPU"
    assert gx.devices() == {gpu} and gy.devices() == {gpu}, "gradients left the GPU"
    return tuple(vals), (gx, gy)


def _run_warp_cuda(dfn, d):
    f = gb.to_warp(dfn)
    wx = wp.array(d["x"], requires_grad=True, device="cuda:0")
    wy = wp.array(d["y"], requires_grad=True, device="cuda:0")
    tape = wp.Tape()
    with tape:
        o1, o2 = f(wx, wy)
    assert str(o1.device) == "cuda:0" and str(o2.device) == "cuda:0", "outputs left the GPU"
    # Seeding the outputs with the cotangents makes tape.backward compute the VJP directly.
    tape.backward(grads={o1: wp.array(d["c1"], device="cuda:0"), o2: wp.array(d["c2"], device="cuda:0")})
    assert str(wx.grad.device) == "cuda:0" and str(wy.grad.device) == "cuda:0", "gradients left the GPU"
    return (o1, o2), (wx.grad, wy.grad)


def _run_tf_gpu(dfn, d, graph=False):
    f = gb.to_tf(dfn)

    def step(tx, ty):
        with tf.GradientTape() as tape:
            tape.watch([tx, ty])
            o1, o2 = f(tx, ty)
            loss = tf.reduce_sum(o1 * d["c1"]) + tf.reduce_sum(o2 * d["c2"])
        gx, gy = tape.gradient(loss, [tx, ty])
        return o1, o2, gx, gy

    with tf.device("/GPU:0"):
        if graph:
            spec = tf.TensorSpec([3], tf.as_dtype(d["x"].dtype))
            step = tf.function(step, input_signature=[spec, spec])
        o1, o2, gx, gy = step(tf.constant(d["x"]), tf.constant(d["y"]))
    for t in (o1, o2, gx, gy):
        assert "GPU" in t.device, f"TF result left the GPU: {t.device}"
    return (o1, o2), (gx, gy)


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("source", GPU_SOURCES)
def test_torch_cuda_consumer(source, gdata, dtype):
    require_gpu("torch", source)
    _check(*_run_torch_cuda(gpu_source(source), gdata), gdata, dtype)


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_torch_cuda_cache_residuals(source, gdata, dtype):
    require_gpu("torch", source)
    _check(*_run_torch_cuda(gpu_source(source), gdata, cache_residuals=True), gdata, dtype)


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_jax_gpu_consumer(source, gdata, dtype):
    require_gpu("jax", source)
    _check(*_run_jax_gpu(gpu_source(source), gdata), gdata, dtype)


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_jax_gpu_jit(source, gdata, dtype):
    require_gpu("jax", source)
    _check(*_run_jax_gpu(gpu_source(source), gdata, jit=True), gdata, dtype)


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_tf_gpu_consumer(source, gdata, dtype):
    require_gpu("tf", source)
    _check(*_run_tf_gpu(gpu_source(source), gdata), gdata, dtype)


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_tf_function_gpu_consumer(source, gdata, dtype):
    require_gpu("tf", source)
    _check(*_run_tf_gpu(gpu_source(source), gdata, graph=True), gdata, dtype)


@pytest.mark.parametrize("source", GPU_SOURCES)
def test_warp_cuda_consumer(source, gdata, dtype):
    require_gpu("warp", source)
    _check(*_run_warp_cuda(gpu_source(source), gdata), gdata, dtype)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize("source", GPU_SOURCES)
def test_torch_compile_cuda(source, backend, gdata, dtype):
    require_gpu("torch", source)
    torch._dynamo.reset()
    f = gb.to_torch(gpu_source(source))
    c1, c2 = torch.tensor(gdata["c1"], device="cuda"), torch.tensor(gdata["c2"], device="cuda")

    @torch.compile(fullgraph=True, backend=backend)
    def loss_fn(x, y):
        o1, o2 = f(x, y)
        return (o1 * c1).sum() + (o2 * c2).sum(), o1, o2

    tx = torch.tensor(gdata["x"], device="cuda", requires_grad=True)
    ty = torch.tensor(gdata["y"], device="cuda", requires_grad=True)
    loss, o1, o2 = loss_fn(tx, ty)
    loss.backward()
    _check((o1.detach(), o2.detach()), (tx.grad, ty.grad), gdata, dtype)


# ----------------------------------------------------------------------------- execution-device probes
def test_warp_source_runs_on_cuda_kernel(gdata, dtype):
    """A Warp source called with a CUDA torch tensor must launch its kernel on cuda."""
    require_gpu("torch", "warp")
    seen = []

    def fn(x, y):
        seen.append(str(x.device))
        return _warp_fn(x, y)

    _check(*_run_torch_cuda(gb.from_warp(fn), gdata), gdata, dtype)
    assert seen and all("cuda" in dev for dev in seen), f"Warp ran on {set(seen)}"


@pytest.mark.parametrize("consumer", ["torch", "jax", "warp"])
def test_tf_source_executes_on_gpu(consumer, gdata, dtype):
    """A TensorFlow source called from another framework must run its ops on the GPU."""
    require_gpu("tf", consumer)
    seen = []

    def fn(x, y):
        # Symbolic tensors from the shape-inference trace have no device; only eager calls execute ops.
        seen.append((tf.executing_eagerly(), x.device))
        return tf.sin(x) * y, tf.reshape(tf.reduce_sum(x**2), [1])

    run = {"torch": _run_torch_cuda, "jax": _run_jax_gpu, "warp": _run_warp_cuda}[consumer]
    _check(*run(gb.from_tf(fn, name="F_tf_probe"), gdata), gdata, dtype)
    eager = [dev for is_eager, dev in seen if is_eager]
    assert eager and all("GPU" in dev for dev in eager), f"TF source ran on: {set(eager)}"


# ----------------------------------------------------------------------------- regressions
def test_misaligned_gpu_view_into_jax(dtype):
    """A detached CUDA view starting mid-allocation goes to XLA zero-copy and must be realigned.

    torch.func.vmap hands sources exactly such views (x.select(0, b)); XLA needs 16-byte alignment.
    Tensors that require grad never hit this: PyTorch refuses to export them over DLPack.
    """
    require_gpu("torch", "jax")
    full = torch.arange(1, 10, dtype=getattr(torch, dtype), device="cuda")
    view = full[1:]
    assert view.data_ptr() % 16 and not view.requires_grad, "test setup: must exercise the zero-copy path"
    f = gb.to_torch(gb.from_jax(lambda x: jnp.sin(x) * 2))
    out = f(view)
    ref = 2 * np.sin(np.arange(2, 10, dtype=np.float64))
    assert out.device.type == "cuda" and out.dtype == view.dtype
    np.testing.assert_allclose(_np(out), ref, **TOLS[dtype])
    # The view must not have been modified by the realignment copy.
    np.testing.assert_array_equal(_np(full), np.arange(1, 10, dtype=dtype))


# ----------------------------------------------------------------------------- zero-copy
def _cuda_memcpys(fn):
    """Count device<->host memcpys issued by any framework (CUPTI sees the whole process)."""
    from torch.profiler import ProfilerActivity, profile

    fn()  # warm-up: compilation and caches
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Profiler clears events")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
    names = [e.name for e in prof.events() if "Memcpy" in e.name]
    return sum("DtoH" in n for n in names), sum("HtoD" in n for n in names)


ZERO_COPY_PAIRS = [
    ("jax", "torch", False),
    ("jax", "torch", True),
    ("warp", "torch", False),
    ("warp", "torch", True),
    ("torch", "warp", False),
    ("jax", "warp", False),
    ("warp", "warp", False),
    ("tf", "torch", False),
    ("tf", "torch", True),
    ("tf", "warp", False),
    ("torch", "jax", False),
    ("tf", "jax", False),
    ("warp", "jax", False),
    ("jax", "jax", False),
    ("torch", "tf", False),
    ("jax", "tf", False),
    ("warp", "tf", False),
    ("tf", "tf", False),
]


@pytest.mark.parametrize("source,consumer,cache_residuals", ZERO_COPY_PAIRS)
def test_zero_copy_on_gpu(source, consumer, cache_residuals, dtype):
    """Forward and backward stay on the GPU: no device<->host copy, and correct gradients."""
    require_gpu(source, consumer)
    n = 1 << 16
    rng = np.random.default_rng(0)
    x_np, y_np = rng.standard_normal(n).astype(dtype), rng.standard_normal(n).astype(dtype)

    if consumer == "torch":
        f = gb.to_torch(gpu_source(source), cache_residuals=cache_residuals)
        x = torch.tensor(x_np, device="cuda", requires_grad=True)
        y = torch.tensor(y_np, device="cuda", requires_grad=True)

        def step():
            x.grad = y.grad = None
            o1, o2 = f(x, y)
            (o1.sum() + o2.sum()).backward()

        get_grad = lambda: x.grad  # noqa: E731
    elif consumer == "jax":
        f = gb.to_jax(gpu_source(source))
        x, y = jax.device_put(x_np, gpu_device()), jax.device_put(y_np, gpu_device())
        # Under jit, as JAX code runs in practice; an un-jitted grad retraces on every call.
        grad_fn = jax.jit(jax.grad(lambda x, y: sum(jnp.sum(o) for o in f(x, y)), argnums=0))
        result = {}

        def step():
            result["g"] = jax.block_until_ready(grad_fn(x, y))

        get_grad = lambda: result["g"]  # noqa: E731
    elif consumer == "tf":
        f = gb.to_tf(gpu_source(source))
        with tf.device("/GPU:0"):
            x, y = tf.identity(tf.constant(x_np)), tf.identity(tf.constant(y_np))
        result = {}

        def step():
            with tf.device("/GPU:0"):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    loss = sum(tf.reduce_sum(o) for o in f(x, y))
                result["g"] = tape.gradient(loss, x)
            # Wait for TF's work without copying the gradient to the host inside the profile.
            tf.test.experimental.sync_devices()

        get_grad = lambda: result["g"]  # noqa: E731
    else:
        f = gb.to_warp(gpu_source(source))
        x = wp.array(x_np, device="cuda:0", requires_grad=True)
        y = wp.array(y_np, device="cuda:0", requires_grad=True)
        ones_n, ones_1 = wp.ones(n, dtype=x.dtype, device="cuda:0"), wp.ones(1, dtype=x.dtype, device="cuda:0")

        def step():
            x.grad.zero_()
            y.grad.zero_()
            tape = wp.Tape()
            with tape:
                o1, o2 = f(x, y)
            tape.backward(grads={o1: ones_n, o2: ones_1})

        get_grad = lambda: x.grad  # noqa: E731

    with warnings.catch_warnings():
        warnings.simplefilter("error", gb.HostCopyWarning)
        dtoh, htod = _cuda_memcpys(step)
    assert (dtoh, htod) == (0, 0), f"{dtoh} device->host and {htod} host->device copies"
    x64, y64 = x_np.astype(np.float64), y_np.astype(np.float64)
    np.testing.assert_allclose(_np(get_grad()), np.cos(x64) * y64 + 2 * x64, **TOLS[dtype])


@pytest.mark.parametrize("direction", ["warp->torch", "torch->warp"])
def test_large_zero_copy_output_on_side_stream(direction):
    """A consumer on a non-default torch stream reading a 16M-element zero-copy result sees final values.

    Zero-copy hands over a pointer while the producer's kernel may still be running on its own
    stream; DLPack's stream handshake is what orders the consumer after it. ``x + 1`` on integers
    below 2**24 is exact in float32, so any mismatch is stale data, not rounding (Warp may fuse
    ``a*b + c`` into one FMA where PyTorch rounds twice). A non-default stream is used because
    PyTorch's default is CUDA's legacy stream, which waits on Warp's stream implicitly.

    Not a reliable race detector: with Warp 1.19 on an RTX 5090, the unordered ``wp.to_torch`` also
    passed 300/300 trials. It guards correctness of the large-array, side-stream path.
    """
    require_gpu("torch", "warp")
    n = 1 << 24

    @wp.kernel
    def _affine(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = x[i] + 1.0

    @wp.kernel
    def _slow_affine(x: wp.array(dtype=wp.float32), iters: int, out: wp.array(dtype=wp.float32)):
        # Same exact result as _affine, but slow enough (runtime-length loop) to open a race window.
        i = wp.tid()
        acc = float(0.0)
        for _k in range(iters):
            acc = acc + 1.0
        out[i] = x[i] + acc / float(iters)

    @wp.kernel
    def _absdiff(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        wp.atomic_add(out, 0, wp.abs(a[i] - b[i]))

    def warp_affine(x):
        out = wp.empty(x.shape, dtype=wp.float32, requires_grad=True, device=x.device)
        wp.launch(_slow_affine, dim=x.shape[0], inputs=[x, 1024], outputs=[out], device=x.device)
        return out

    side = torch.cuda.Stream()
    if direction == "warp->torch":
        f = gb.to_torch(gb.from_warp(warp_affine))
        x = torch.arange(n, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        with torch.cuda.stream(side):
            for _ in range(3):
                out = f(x)
                assert torch.equal(out, x + 1.0)
    else:
        f = gb.to_warp(gb.from_torch(lambda t: t + 1.0))
        x = wp.array(np.arange(n, dtype=np.float32), device="cuda:0")
        check = wp.empty_like(x)
        wp.launch(_affine, dim=n, inputs=[x], outputs=[check], device="cuda:0")
        wp.synchronize()
        with torch.cuda.stream(side):
            for _ in range(3):
                out = f(x)
                # Read on Warp's stream: must observe torch's finished write.
                diff = wp.zeros(1, dtype=wp.float32, device="cuda:0")
                wp.launch(_absdiff, dim=n, inputs=[out, check], outputs=[diff], device="cuda:0")
                assert diff.numpy()[0] == 0.0


def test_misaligned_warp_view_into_jax(dtype):
    """Warp arrays now reach JAX zero-copy too, so a mid-allocation Warp view must be realigned."""
    require_gpu("warp", "jax")
    full = wp.array(np.arange(1, 10, dtype=dtype), device="cuda:0", requires_grad=True)
    view = full[1:]
    assert view.ptr % 16, "test setup: the view must be misaligned"
    f = gb.to_warp(gb.from_jax(lambda x: jnp.sin(x) * 2))
    tape = wp.Tape()
    with tape:
        out = f(view)
    tape.backward(grads={out: wp.ones(out.shape, dtype=out.dtype, device="cuda:0")})
    ref = np.arange(2, 10, dtype=np.float64)
    np.testing.assert_allclose(_np(out), 2 * np.sin(ref), **TOLS[dtype])
    np.testing.assert_allclose(_np(view.grad), 2 * np.cos(ref), **TOLS[dtype])


@pytest.mark.parametrize("cache_residuals", [False, True])
def test_inplace_change_between_forward_and_backward_is_caught(cache_residuals):
    """JAX now aliases the torch input; PyTorch's saved-tensor version check must still fire."""
    require_gpu("torch", "jax")
    f = gb.to_torch(gpu_source("jax"), cache_residuals=cache_residuals)
    leaf = torch.randn(8, device="cuda", requires_grad=True)
    x = leaf * 1.0
    o1, o2 = f(x, torch.ones(8, device="cuda"))
    with torch.no_grad():
        x.add_(1.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        (o1.sum() + o2.sum()).backward()


# ----------------------------------------------------------------------------- Warp tape ownership
def test_to_warp_records_only_the_bridge_on_the_callers_tape():
    """The bridged function is opaque: its own kernels must not be replayed by the caller's tape.

    device= keeps the source on the bridged path (without it, Warp-from-Warp composes natively).
    """
    require_gpu("warp")
    f = gb.to_warp(gb.from_warp(_warp_fn, device="cuda:0"))
    x = wp.array(np.linspace(0.1, 1.0, 8), device="cuda:0", requires_grad=True)
    y = wp.array(np.ones(8), device="cuda:0", requires_grad=True)
    tape = wp.Tape()
    with tape:
        f(x, y)
    assert len(tape.launches) == 1, f"caller's tape recorded {len(tape.launches)} entries, expected 1"


@pytest.mark.parametrize("source", ["warp", "torch", "jax"])
def test_to_warp_gradient_adds_to_existing_grad(source):
    """x feeds a native Warp kernel and the bridge on one tape; both gradient contributions must survive.

    The bridge's backward runs after the native kernel's, so x.grad already holds a value when the
    bridge zeroes its internal tape and accumulates its own contribution.
    """
    require_gpu("warp", source)

    @wp.kernel
    def _double(x: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        out[i] = wp.float64(2.0) * x[i]

    f = gb.to_warp(gpu_source(source))
    x_np = np.linspace(0.1, 1.0, 8)
    x = wp.array(x_np, device="cuda:0", requires_grad=True)
    y = wp.array(np.ones(8), device="cuda:0", requires_grad=True)
    doubled = wp.zeros(8, dtype=wp.float64, device="cuda:0", requires_grad=True)
    tape = wp.Tape()
    with tape:
        o1, o2 = f(x, y)  # recorded first, so its backward runs last
        wp.launch(_double, dim=8, inputs=[x], outputs=[doubled], device="cuda:0")
    ones8 = wp.ones(8, dtype=wp.float64, device="cuda:0")
    tape.backward(grads={o1: ones8, o2: wp.ones(1, dtype=wp.float64, device="cuda:0"), doubled: ones8})
    # d/dx [sin(x) * 1 + sum(x^2) + 2x] = cos(x) + 2x + 2
    np.testing.assert_allclose(x.grad.numpy(), np.cos(x_np) + 2 * x_np + 2.0, rtol=1e-9)


def test_misaligned_gpu_view_into_tf(dtype):
    """TensorFlow's Eigen kernels abort the process on data not aligned to 64 bytes; views are realigned.

    Elementwise ops tolerate misalignment; the reduction's gradient (a ``tile``) is what checks it.
    """
    require_gpu("torch", "tf")
    full = torch.arange(1, 10, dtype=getattr(torch, dtype), device="cuda", requires_grad=True)
    view = full.detach()[1:]
    assert view.data_ptr() % 64, "test setup: the view must be misaligned"
    view.requires_grad_(True)
    seen = []

    def fn(x):
        seen.append(x.device)
        return tf.sin(x) * tf.reduce_sum(x)

    f = gb.to_torch(gb.from_tf(fn))
    out = f(view)
    out.sum().backward()
    x = np.arange(2, 10, dtype=np.float64)
    assert out.device.type == "cuda" and out.dtype == view.dtype
    np.testing.assert_allclose(_np(out), np.sin(x) * x.sum(), **TOLS[dtype])
    np.testing.assert_allclose(_np(view.grad), np.cos(x) * x.sum() + np.sin(x).sum(), **TOLS[dtype])
    assert any("GPU" in d for d in seen), f"TF ran on {set(seen)}"


@pytest.mark.parametrize("cache_residuals", [None, False, True])
def test_buffer_reusing_warp_source(cache_residuals):
    """Simulators reuse preallocated state buffers; here several calls share one before one backward.

    Every mode must give exact gradients: recomputation by construction, caching because a later
    call touching the shared buffer marks earlier cached tapes stale. The adjoint of _square_out
    reads the shared buffer, so a stale cached tape would give wrong gradients.
    """
    require_gpu("torch", "warp")
    state = wp.zeros(4, dtype=wp.float32, device="cuda:0", requires_grad=True)  # shared across calls

    @wp.kernel
    def _square_into(x: wp.array(dtype=wp.float32), s: wp.array(dtype=wp.float32)):
        i = wp.tid()
        s[i] = x[i] * x[i]

    @wp.kernel
    def _square_out(s: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = s[i] * s[i]  # its adjoint reads s: the shared buffer

    def sim(x):
        wp.launch(_square_into, dim=4, inputs=[x], outputs=[state], device=x.device)
        out = wp.empty(4, dtype=wp.float32, requires_grad=True, device=x.device)
        wp.launch(_square_out, dim=4, inputs=[state], outputs=[out], device=x.device)
        return out

    f = gb.to_torch(gb.from_warp(sim), cache_residuals=cache_residuals)
    xs = [torch.full((4,), v, device="cuda", requires_grad=True) for v in (1.0, 2.0, 3.0)]
    sum(f(x).sum() for x in xs).backward()
    assert [x.grad[0].item() for x in xs] == [4.0, 32.0, 108.0]  # d/dx x^4 = 4x^3


# ----------------------------------------------------------------------------- JAX consumer via XLA FFI
def test_jax_ffi_large_output_exact():
    """16M elements through the FFI path, repeated: values must be final and exact on every call.

    Exercises the stream handshakes (torch reads XLA's buffers, XLA receives torch's result) and the
    end-of-call wait that keeps the result alive until XLA's stream has copied it.
    """
    require_gpu("torch", "jax")
    n = 1 << 24
    f = jax.jit(gb.to_jax(gb.from_torch(lambda t: t + 1.0)))
    x = jax.device_put(jnp.arange(n, dtype=jnp.float32), gpu_device())
    for _ in range(5):
        out = f(x)
        assert bool(jnp.array_equal(out, x + 1.0))


def test_jax_consumer_scalar_output_falls_back(dtype):
    """Warp arrays cannot be 0-d, so a scalar output takes the pure_callback path, still correct."""
    require_gpu("torch", "jax")
    f = gb.to_jax(gb.from_torch(lambda t: (t**2).sum()))
    x = jax.device_put(np.linspace(0.1, 1.0, 5).astype(dtype), gpu_device())
    val, g = jax.jit(jax.value_and_grad(f))(x)
    x64 = np.linspace(0.1, 1.0, 5)
    np.testing.assert_allclose(float(val), np.sum(x64**2), **TOLS[dtype])
    np.testing.assert_allclose(np.asarray(g, dtype=np.float64), 2 * x64, **TOLS[dtype])


def test_warp_from_warp_composes_natively():
    """A Warp source called from Warp runs on the caller's tape directly: same kernels, no bridge."""
    require_gpu("warp")
    f = gb.to_warp(gpu_source("warp"))
    x = wp.array(np.linspace(0.1, 1.0, 8), device="cuda:0", requires_grad=True)
    y = wp.array(np.ones(8), device="cuda:0", requires_grad=True)
    tape = wp.Tape()
    with tape:
        o1, o2 = f(x, y)
    assert all(isinstance(e, list) for e in tape.launches), "expected only native kernel launches"
    tape.backward(grads={o1: wp.ones(8, dtype=wp.float64, device="cuda:0"), o2: wp.ones(1, dtype=wp.float64, device="cuda:0")})
    x64 = np.linspace(0.1, 1.0, 8)
    np.testing.assert_allclose(x.grad.numpy(), np.cos(x64) + 2 * x64, rtol=1e-9)


# ----------------------------------------------------------------------------- TF consumer op
def test_tf_consumer_propagates_the_bridged_exception():
    """An error inside the bridged function surfaces as itself, not as TensorFlow's InternalError."""
    require_gpu("torch", "tf")

    def broken(x):
        raise ValueError("boom from the bridged function")

    f = gb.to_tf(gb.from_torch(broken, n_inputs=1))
    with tf.device("/GPU:0"):
        x = tf.ones([4])
    with pytest.raises(Exception, match="boom from the bridged function"):
        f(x)


def test_tf_consumer_falls_back_without_the_op(monkeypatch):
    """Without the compiled op (e.g. no compiler), calls from TensorFlow still work via py_function."""
    require_gpu("torch", "tf")
    from gradbridge import _tf_native

    monkeypatch.setattr(_tf_native, "load", lambda: None)
    f = gb.to_tf(gb.from_torch(lambda a, b: torch.sin(a) * b))
    with tf.device("/GPU:0"):
        x, y = tf.constant(np.linspace(0.1, 1.0, 5)), tf.constant(np.full(5, 2.0))
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = f(x, y)
        g = tape.gradient(out, x)
    x64 = np.linspace(0.1, 1.0, 5)
    np.testing.assert_allclose(out.numpy(), 2 * np.sin(x64), rtol=1e-12)
    np.testing.assert_allclose(g.numpy(), 2 * np.cos(x64), rtol=1e-12)
