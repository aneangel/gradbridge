"""``from_warp(..., capture=True)``: CUDA-graph replays must give exactly the eager results.

A replay reuses fixed buffers, so every test targets a way that could leak between calls: two
forwards before one backward, a reused residual, held outputs, shared buffers written by another
call, a capture that fails, and consumers that run on a foreign stream.
"""

import warnings

import numpy as np
import pytest
import torch
import warp as wp

import gradbridge as gb

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and wp.get_cuda_device_count() > 0), reason="needs CUDA for PyTorch and Warp"
)

STEPS = 12


@wp.kernel
def _step(x: wp.array(dtype=wp.float64), y_in: wp.array(dtype=wp.float64), y_out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y_out[i] = wp.sin(y_in[i]) * x[i] + wp.float64(0.5) * y_in[i]


def _rollout(x):
    """STEPS launches with STEPS intermediates: what the backward reads, a replay overwrites."""
    ys = [wp.zeros_like(x, requires_grad=True) for _ in range(STEPS + 1)]
    wp.launch(_step, dim=x.shape[0], inputs=[x, x], outputs=[ys[0]], device=x.device)
    for k in range(STEPS):
        wp.launch(_step, dim=x.shape[0], inputs=[x, ys[k]], outputs=[ys[k + 1]], device=x.device)
    return ys[-1]


def _pair(**kwargs):
    eager = gb.to_torch(gb.from_warp(_rollout), **kwargs)
    graphed = gb.to_torch(gb.from_warp(_rollout, capture=True), **kwargs)
    return eager, graphed


def _x(seed, n=257):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(n, device="cuda", dtype=torch.float64, generator=g) + 0.5


def _value_and_grad(f, x, weight):
    x = x.clone().requires_grad_(True)
    y = f(x)
    (y * weight).sum().backward()
    return y.detach(), x.grad


@pytest.fixture
def replays(monkeypatch):
    """Counts graph launches, so a test cannot pass by never using a graph."""
    count = [0]
    launch = wp.capture_launch

    def counting(*args, **kwargs):
        count[0] += 1
        return launch(*args, **kwargs)

    monkeypatch.setattr(wp, "capture_launch", counting)
    return count


def test_replays_match_eager_bit_for_bit(replays):
    """Call 1 runs eagerly, call 2 captures, calls 3+ replay; every one equals the eager path."""
    eager, graphed = _pair()
    for call in range(5):
        x, w = _x(call), _x(100 + call)
        ye, ge = _value_and_grad(eager, x, w)
        yg, gg = _value_and_grad(graphed, x, w)
        assert torch.equal(ye, yg), f"value differs on call {call}"
        assert torch.equal(ge, gg), f"gradient differs on call {call}"
    # call 2: forward replay (its backward runs eagerly once, then is captured); calls 3-5: both
    assert replays[0] == 1 + 2 * 3


def test_two_forwards_before_one_backward():
    """The second replay overwrites the first's intermediates; the first's backward must recompute.

    The reference is one clean forward and backward per input, so it does not depend on the
    invalidation logic under test.
    """
    eager, graphed = _pair()
    for _ in range(2):  # warm: the graph exists before the two-forward step
        _value_and_grad(graphed, _x(0), _x(1))
    ref_a, ref_b = _value_and_grad(eager, _x(2), _x(4))[1], _value_and_grad(eager, _x(3), _x(5))[1]
    a, b = _x(2).requires_grad_(True), _x(3).requires_grad_(True)
    (graphed(a) * _x(4) + graphed(b) * _x(5)).sum().backward()
    assert torch.equal(a.grad, ref_a)
    assert torch.equal(b.grad, ref_b)


def test_retain_graph_reuses_the_graph_residual():
    _, graphed = _pair()
    for _ in range(2):
        _value_and_grad(graphed, _x(0), _x(1))
    x = _x(2).requires_grad_(True)
    out = graphed(x)
    out.sum().backward(retain_graph=True)
    first = x.grad.clone()
    out.sum().backward()
    torch.testing.assert_close(x.grad, 2 * first, rtol=0, atol=0)


def test_held_outputs_and_gradients_are_not_overwritten_by_later_replays():
    _, graphed = _pair()
    held = [_value_and_grad(graphed, _x(i), _x(10 + i)) for i in range(4)]
    snapshot = [(y.clone(), g.clone()) for y, g in held]
    for i in range(4, 8):
        _value_and_grad(graphed, _x(i), _x(10 + i))
    for (y, g), (ys, gs) in zip(held, snapshot):
        assert torch.equal(y, ys) and torch.equal(g, gs)


def test_eval_without_grad_replays_the_graph():
    eager, graphed = _pair()
    for i in range(2):
        _value_and_grad(graphed, _x(i), _x(1))
    with torch.no_grad():
        assert torch.equal(graphed(_x(7)), eager(_x(7)))


def test_inference_between_forward_and_backward(replays):
    """Inference replays the graph and invalidates the pending training residual."""
    eager, graphed = _pair()
    for i in range(2):
        _value_and_grad(graphed, _x(i), _x(1))
    ref = _value_and_grad(eager, _x(2), _x(4))[1]
    x = _x(2).requires_grad_(True)
    out = graphed(x)
    before = replays[0]
    with torch.no_grad():
        graphed(_x(7))
    assert replays[0] == before + 1
    (out * _x(4)).sum().backward()
    assert torch.equal(x.grad, ref)


def test_new_shape_gets_its_own_graph():
    eager, graphed = _pair()
    for n in (64, 64, 64, 300, 300, 300, 64):
        x, w = _x(n, n), _x(n + 1, n)
        assert torch.equal(_value_and_grad(eager, x, w)[1], _value_and_grad(graphed, x, w)[1])


# Two functions that share a preallocated buffer, like a simulator's state arrays.
_SHARED = {}


def _shared():
    if "buf" not in _SHARED:
        _SHARED["buf"] = wp.zeros(257, dtype=wp.float64, device="cuda", requires_grad=True)
    return _SHARED["buf"]


def _through_shared(x):
    buf = _shared()
    wp.launch(_step, dim=x.shape[0], inputs=[x, x], outputs=[buf], device=x.device)
    y = wp.zeros_like(x, requires_grad=True)
    wp.launch(_step, dim=x.shape[0], inputs=[x, buf], outputs=[y], device=x.device)
    return y


def _overwrite_shared(x):
    y = wp.zeros_like(x, requires_grad=True)
    wp.launch(_step, dim=x.shape[0], inputs=[x, x], outputs=[_shared()], device=x.device)
    wp.launch(_step, dim=x.shape[0], inputs=[x, _shared()], outputs=[y], device=x.device)
    return y


def test_captured_residual_invalidated_by_another_call_writing_shared_memory():
    """The reference runs with no call in between, so it does not depend on the invalidation logic."""
    eager = gb.to_torch(gb.from_warp(_through_shared))
    graphed = gb.to_torch(gb.from_warp(_through_shared, capture=True))
    other = gb.to_torch(gb.from_warp(_overwrite_shared))
    for _ in range(2):
        _value_and_grad(graphed, _x(0), _x(1))
    ref = _value_and_grad(eager, _x(2), _x(4))[1]
    x = _x(2).requires_grad_(True)
    y = graphed(x)
    other(_x(3) * 3)  # overwrites the shared buffer this residual's backward reads
    (y * _x(4)).sum().backward()
    assert torch.equal(x.grad, ref)


def _syncs_on_host(x):
    float(x.numpy()[0])  # a host read: not allowed while capturing
    return _rollout(x)


def test_failed_capture_warns_and_falls_back(replays):
    eager = gb.to_torch(gb.from_warp(_rollout))
    graphed = gb.to_torch(gb.from_warp(_syncs_on_host, capture=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for call in range(4):
            x, w = _x(call), _x(9)
            assert torch.equal(_value_and_grad(eager, x, w)[1], _value_and_grad(graphed, x, w)[1])
    assert sum("capture failed" in str(w.message) for w in caught) == 1
    assert replays[0] == 0
    torch.cuda.synchronize()  # the device is still healthy after the aborted capture


def test_cpu_inputs_run_eagerly():
    f = gb.to_torch(gb.from_warp(_rollout, capture=True))
    for call in range(3):
        x = _x(call).cpu().requires_grad_(True)
        f(x).sum().backward()
        assert x.grad.device.type == "cpu" and torch.isfinite(x.grad).all()


def test_jax_consumer_replays_on_xlas_stream(replays):
    """Inside JAX, the graphs are captured and replayed on XLA's own stream, matching eager."""
    jax = pytest.importorskip("jax")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU for JAX")
    jax.config.update("jax_enable_x64", True)
    graphed = gb.to_jax(gb.from_warp(_rollout, capture=True))
    eager = gb.to_jax(gb.from_warp(_rollout))
    w = jax.numpy.asarray(np.linspace(0.5, 1.5, 257))
    for call in range(4):
        x = jax.numpy.asarray(np.linspace(0.6, 1.4, 257) + 0.01 * call)
        ge = jax.jit(jax.grad(lambda x: (eager(x) * w).sum()))(x)
        gg = jax.jit(jax.grad(lambda x: (graphed(x) * w).sum()))(x)
        np.testing.assert_array_equal(np.asarray(ge), np.asarray(gg))
    assert replays[0] > 0


def test_capture_status_of_a_foreign_stream():
    """The guard that keeps capture=True out of someone else's graph capture sees PyTorch's."""
    from gradbridge._convert import stream_is_capturing

    side = torch.cuda.Stream()
    assert not stream_is_capturing(side.cuda_stream)
    graph, buf = torch.cuda.CUDAGraph(), torch.zeros(1, device="cuda")
    with torch.cuda.graph(graph):
        inside = stream_is_capturing(torch.cuda.current_stream().cuda_stream)
        buf += 1  # a non-empty graph
    assert inside


@wp.kernel
def _accumulate(x: wp.array(dtype=wp.float64), acc: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(acc, i, wp.sin(x[i]) * x[i])


_STATE = {}


def _through_state(x):
    """A buffer that lives across calls and is accumulated into (a simulator's forces): cleared by
    an unrecorded memset, then atomically added to, so its gradient is not consumed by backward."""
    if "acc" not in _STATE:
        _STATE["acc"] = wp.zeros(x.shape[0], dtype=wp.float64, device=x.device, requires_grad=True)
    acc = _STATE["acc"]
    acc.zero_()
    wp.launch(_accumulate, dim=x.shape[0], inputs=[x], outputs=[acc], device=x.device)
    y = wp.zeros_like(x, requires_grad=True)
    wp.launch(_step, dim=x.shape[0], inputs=[x, acc], outputs=[y], device=x.device)
    return y


def test_preallocated_buffers_do_not_carry_gradients_between_calls(replays):
    """Every call, eager first and replayed after, equals a fresh single call on fresh buffers."""
    refs = []
    for call in range(4):
        _STATE.clear()
        refs.append(_value_and_grad(gb.to_torch(gb.from_warp(_through_state)), _x(call), _x(9))[1])
    _STATE.clear()
    graphed = gb.to_torch(gb.from_warp(_through_state, capture=True))
    for call in range(4):
        assert torch.equal(_value_and_grad(graphed, _x(call), _x(9))[1], refs[call]), f"call {call}"
    assert replays[0] > 0
