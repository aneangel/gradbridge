"""Per-call cost of a gradbridge forward+backward, against native code and hand-written bridges.

The benchmarked function is elementwise, ``F(x, y) = sin(x) * y``, so the kernel itself is
negligible and the timings isolate the bridge. Each measurement:

* warms up (compilation, kernel loading, allocator growth are excluded),
* synchronizes every framework before stopping the clock,
* reports the median over repeated calls (at least 5, at most 200, about 1 s of work).

Usage::

    python benchmarks/bench.py                       # all pairs, default sizes
    python benchmarks/bench.py --sizes 1000 1000000  # custom sizes
    python benchmarks/bench.py --markdown out.md     # also write a markdown report
"""

import argparse
import os
import platform
import statistics
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# PyTorch first: see the TensorFlow/Triton LLVM note in gradbridge/torch.py.
import gradbridge.torch  # noqa: F401, E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import gradbridge as gb  # noqa: E402

wp.config.quiet = True
wp.init()
SOURCES = ["numpy", "torch", "jax", "tf", "warp"]
CONSUMERS = ["torch", "jax", "tf", "warp"]


# ----------------------------------------------------------------------------- sources
@wp.kernel
def _sin_mul(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.sin(x[i]) * y[i]


def _warp_fn(x, y):
    out = wp.empty(x.shape, dtype=x.dtype, requires_grad=True, device=x.device)
    wp.launch(_sin_mul, dim=x.shape[0], inputs=[x, y], outputs=[out], device=x.device)
    return out


def make_source(name):
    if name == "numpy":
        return gb.from_numpy(lambda x, y: np.sin(x) * y, vjp=lambda x, y, c: (c * np.cos(x) * y, c * np.sin(x)))
    if name == "torch":
        return gb.from_torch(lambda x, y: torch.sin(x) * y)
    if name == "jax":
        return gb.from_jax(lambda x, y: jnp.sin(x) * y)
    if name == "tf":
        return gb.from_tf(lambda x, y: tf.sin(x) * y)
    return gb.from_warp(_warp_fn)


# ----------------------------------------------------------------------------- consumer steps
def torch_step(f, n):
    """One forward+backward from PyTorch; returns a callable that runs it to completion."""
    x, y, c = (torch.randn(n, device="cuda", requires_grad=k != "c") for k in "xyc")

    def step():
        x.grad = y.grad = None
        f(x, y).backward(c)
        torch.cuda.synchronize()

    return step


def jax_step(f, n):
    gpu = [d for d in jax.devices() if d.platform == "gpu"][0]
    x, y, c = (jax.device_put(np.random.default_rng(i).standard_normal(n).astype(np.float32), gpu) for i in range(3))
    grad = jax.jit(jax.grad(lambda x, y: jnp.sum(f(x, y) * c), argnums=(0, 1)))
    return lambda: jax.block_until_ready(grad(x, y))


def tf_step(f, n):
    with tf.device("/GPU:0"):
        x, y, c = (tf.identity(tf.random.normal([n])) for _ in range(3))

    def step():
        with tf.device("/GPU:0"):
            with tf.GradientTape() as tape:
                tape.watch([x, y])
                out = f(x, y)
            gx, _ = tape.gradient(out, [x, y], output_gradients=c)
        gx.numpy()  # blocks until TF's work is done (one small device->host copy per step)

    return step


def warp_step(f, n):
    rng = np.random.default_rng(0)
    x, y = (wp.array(rng.standard_normal(n).astype(np.float32), device="cuda:0", requires_grad=True) for _ in "xy")
    c = wp.array(rng.standard_normal(n).astype(np.float32), device="cuda:0")

    def step():
        x.grad.zero_()
        y.grad.zero_()
        tape = wp.Tape()
        with tape:
            out = f(x, y)
        tape.backward(grads={out: c})
        wp.synchronize_device("cuda:0")

    return step


def bridged(source, consumer, n):
    dfn = make_source(source)
    if consumer == "torch":
        return torch_step(gb.to_torch(dfn), n)
    if consumer == "jax":
        return jax_step(gb.to_jax(dfn), n)
    if consumer == "tf":
        return tf_step(gb.to_tf(dfn), n)
    return warp_step(gb.to_warp(dfn), n)


def native(consumer, n):
    """The same function written directly in the consumer framework: the floor for any bridge."""
    if consumer == "torch":
        return torch_step(lambda x, y: torch.sin(x) * y, n)
    if consumer == "jax":
        return jax_step(lambda x, y: jnp.sin(x) * y, n)
    if consumer == "tf":
        return tf_step(lambda x, y: tf.sin(x) * y, n)
    return warp_step(_warp_fn, n)


# ----------------------------------------------------------------------------- hand-written bridges
class _HandWarpToTorch(torch.autograd.Function):
    """The pattern from Warp's PyTorch interop docs: wrap tensors with wp.from_torch, replay a Tape."""

    @staticmethod
    def forward(ctx, x, y):
        ctx.wx, ctx.wy = wp.from_torch(x, requires_grad=True), wp.from_torch(y, requires_grad=True)
        ctx.tape = wp.Tape()
        with wp.ScopedStream(wp.stream_from_torch(x.device)):
            with ctx.tape:
                ctx.wout = _warp_fn(ctx.wx, ctx.wy)
        return wp.to_torch(ctx.wout)

    @staticmethod
    def backward(ctx, grad):
        with wp.ScopedStream(wp.stream_from_torch(grad.device)):
            ctx.tape.backward(grads={ctx.wout: wp.from_torch(grad.contiguous())})
        gx, gy = wp.to_torch(ctx.wx.grad).clone(), wp.to_torch(ctx.wy.grad).clone()
        ctx.tape.zero()
        return gx, gy


class _HandJaxToTorch(torch.autograd.Function):
    """DLPack in both directions plus jax.vjp: the usual jax2torch-style bridge."""

    @staticmethod
    def forward(ctx, x, y):
        jx, jy = jax.dlpack.from_dlpack(x.detach()), jax.dlpack.from_dlpack(y.detach())
        out, ctx.vjp = jax.vjp(lambda a, b: jnp.sin(a) * b, jx, jy)
        return torch.from_dlpack(out)

    @staticmethod
    def backward(ctx, grad):
        gx, gy = ctx.vjp(jax.dlpack.from_dlpack(grad.contiguous()))
        return torch.from_dlpack(gx), torch.from_dlpack(gy)


HAND_WRITTEN = {
    ("warp", "torch"): lambda n: torch_step(_HandWarpToTorch.apply, n),
    ("jax", "torch"): lambda n: torch_step(_HandJaxToTorch.apply, n),
}


# ----------------------------------------------------------------------------- timing
def median_ms(step, budget_s=1.0, min_reps=5, max_reps=200):
    for _ in range(3):
        step()
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and (len(times) < min_reps or time.perf_counter() - start < budget_s):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 100_000, 10_000_000])
    parser.add_argument("--markdown", help="also write the results to this markdown file")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    header = (
        f"{gpu_name}, torch {torch.__version__}, jax {jax.__version__}, tf {tf.__version__}, "
        f"warp {wp.__version__}, Python {platform.python_version()}; float32; median ms per forward+backward"
    )
    lines = [f"# gradbridge benchmark\n\n{header}\n"]
    for n in args.sizes:
        lines.append(f"\n## n = {n:,}\n")
        lines.append("| consumer | native | " + " | ".join(f"from {s}" for s in SOURCES) + " | hand-written |")
        lines.append("|---" * (len(SOURCES) + 3) + "|")
        for c in CONSUMERS:
            row = [f"{median_ms(native(c, n)):.3f}"]
            for s in SOURCES:
                row.append(f"{median_ms(bridged(s, c, n)):.3f}")
            hand = [f"{s}: {median_ms(mk(n)):.3f}" for (s, cc), mk in HAND_WRITTEN.items() if cc == c]
            row.append(", ".join(hand) or "—")
            lines.append(f"| **{c}** | " + " | ".join(row) + " |")
            print(lines[-1], flush=True)
    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(report)


if __name__ == "__main__":
    main()
