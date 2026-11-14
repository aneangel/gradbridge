"""JAX-to-PyTorch callback cost: native GPU buffers versus the previous host fallback.

Each case computes a value and gradient, verifies both paths agree, warms up compilation,
then reports median synchronized ms/call over alternating rounds.
"""

import argparse
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import gradbridge.torch  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import torch

import gradbridge as gb


def make_step(case, x, *, host):
    def source(t):
        return (t * t).sum() if case == "scalar loss" else t * t

    f = gb.to_jax(gb.from_torch(source, batched=case == "batched"))
    call = jax.vmap(f) if case == "batched" else f
    step = jax.jit(jax.value_and_grad(lambda x: jnp.sum(call(x))))
    if host:
        # Force exactly the fallback these signatures used before the native path supported them.
        with (
            patch("gradbridge.jax._native_callable", return_value=None),
            patch("gradbridge.jax._gpu_callable", return_value=None),
        ):
            return step.lower(x).compile()
    return step.lower(x).compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    if args.elements < 256 or args.elements % 256 or min(args.rounds, args.calls) < 1:
        parser.error("--elements must be a positive multiple of 256; --rounds and --calls must be positive")
    torch.set_num_threads(1)
    gpu = jax.devices("gpu")[0]
    shapes = {
        "scalar loss": (args.elements,),
        "batched": (16, args.elements // 16),
        "five dimensions": (4, 4, 4, 4, args.elements // 256),
    }
    lines = [
        "# Native JAX callback benchmark\n",
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, JAX {jax.__version__}; "
        f"{args.elements:,} float32 elements; value + gradient; synchronized ms/call.\n",
        "| Case | Previous host path | Native GPU path | Speedup |",
        "|---|---:|---:|---:|",
    ]
    for case, shape in shapes.items():
        x = jax.device_put(np.linspace(0.1, 1, args.elements, dtype=np.float32).reshape(shape), gpu)
        paths = {"host": make_step(case, x, host=True), "native": make_step(case, x, host=False)}
        previous, native = (jax.block_until_ready(step(x)) for step in paths.values())
        for a, b in zip(previous, native):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        samples = {name: [] for name in paths}
        for _ in range(3):
            for step in paths.values():
                jax.block_until_ready(step(x))
        for i in range(args.rounds):
            for name in list(paths)[:: 1 if i % 2 == 0 else -1]:
                start = time.perf_counter()
                for _ in range(args.calls):
                    jax.block_until_ready(paths[name](x))
                samples[name].append((time.perf_counter() - start) * 1e3 / args.calls)
        times = {name: statistics.median(values) for name, values in samples.items()}
        lines.append(
            f"| {case} | {times['host']:.3f} | {times['native']:.3f} | {times['host'] / times['native']:.2f}× |"
        )
        print(lines[-1], flush=True)
    report = "\n".join(lines) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(report)


if __name__ == "__main__":
    main()
