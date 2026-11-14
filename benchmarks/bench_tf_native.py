"""Native JAX-to-TensorFlow XLA graph versus the standard device callback.

Both evaluate sum(sin(x)**2 * x + 1) and its gradient on the GPU. The native path
compiles the whole TensorFlow function with XLA; the callback path uses tf.function.
Values and gradients are checked, warmup is excluded, and each call is synchronized.
"""

import argparse
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gradbridge.torch  # noqa: E402, F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

import gradbridge as gb  # noqa: E402


def make_step(native):
    f = gb.to_tf(gb.from_jax(lambda x: jnp.sin(x) ** 2), native=native)

    @tf.function(jit_compile=native, autograph=False)
    def step(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(f(x) * x + 1)
        return loss, tape.gradient(loss, x)

    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--markdown")
    parser.add_argument("--hlo", help="write the native graph's optimized HLO")
    args = parser.parse_args()
    if min(args.elements, args.rounds, args.calls) < 1:
        parser.error("--elements, --rounds and --calls must be positive")
    jax.devices("gpu")
    if not torch.cuda.is_available() or not tf.config.list_physical_devices("GPU"):
        parser.error("this benchmark requires PyTorch, JAX and TensorFlow CUDA support")
    with tf.device("/GPU:0"):
        x = tf.linspace(tf.constant(0.1), tf.constant(1.0), args.elements)
        paths = {"callback": make_step(False), "native": make_step(True)}
        reference, lowered = (step(x) for step in paths.values())
        for a, b in zip(reference, lowered):
            assert "GPU:0" in a.device and "GPU:0" in b.device
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)
        for _ in range(3):
            for step in paths.values():
                step(x)
                torch.cuda.synchronize()
        samples = {name: [] for name in paths}
        for i in range(args.rounds):
            for name in list(paths)[:: 1 if i % 2 == 0 else -1]:
                start = time.perf_counter()
                for _ in range(args.calls):
                    paths[name](x)
                    torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1e3 / args.calls)
        times = {name: statistics.median(values) for name, values in samples.items()}
        if args.hlo:
            Path(args.hlo).write_text(paths["native"].experimental_get_compiler_ir(x)(stage="optimized_hlo"))
    report = (
        f"# Native TensorFlow graph benchmark\n\n{torch.cuda.get_device_name(0)}, "
        f"JAX {jax.__version__}, TensorFlow {tf.__version__}; {args.elements:,} float32 elements; "
        f"value + gradient; synchronized median ms/call.\n\n"
        "| Device callback | Native graph + XLA | Speedup |\n|---:|---:|---:|\n"
        f"| {times['callback']:.3f} | {times['native']:.3f} | "
        f"{times['callback'] / times['native']:.2f}× |\n"
    )
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report)


if __name__ == "__main__":
    main()
