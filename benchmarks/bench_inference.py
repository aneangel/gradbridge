"""Inference cost on downstream workloads, with and without backward-state construction.

The recording path is the eager implementation used before inference skipped residuals.
Both paths run under torch.no_grad(), with identical inputs and checked outputs. Compilation
and allocation warmup are excluded; each number is the median of alternating timing rounds.
On CUDA, every call is synchronized before stopping the clock.

Usage::

    pip install -e ".[all,downstream]"
    python benchmarks/bench_inference.py
    python benchmarks/bench_inference.py --device cuda
    python benchmarks/bench_inference.py --workloads newton keras --markdown inference.md
"""

import argparse
import importlib.metadata
import os
import platform
import runpy
import statistics
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch  # noqa: E402

import gradbridge as gb  # noqa: E402
from gradbridge.torch import _eager_function  # noqa: E402


def workload(name, device="cpu"):
    if name == "torchdiffeq":
        import torchdiffeq

        def solve(y0):
            times = torch.tensor([0.0, 1.0], dtype=y0.dtype, device=y0.device)
            return torchdiffeq.odeint(lambda t, y: -0.4 * y, y0, times, method="rk4", options={"step_size": 0.05})[-1]

        return gb.from_torch(solve, device=device), torch.ones(16, device=device, requires_grad=True)

    if name == "diffrax":
        import diffrax
        import jax

        if device == "cuda":
            jax.devices("gpu")  # fail instead of silently timing a CPU fallback

        def solve(y0):
            return diffrax.diffeqsolve(
                diffrax.ODETerm(lambda t, y, args: -0.4 * y),
                diffrax.Tsit5(),
                t0=0.0,
                t1=1.0,
                dt0=0.05,
                y0=y0,
                max_steps=32,
                adjoint=diffrax.DirectAdjoint(),
            ).ys[-1]

        return gb.from_jax(solve), torch.ones(16, device=device, requires_grad=True)

    if name == "keras":
        import keras
        import tensorflow as tf

        target = "/GPU:0" if device == "cuda" else "/CPU:0"
        if device == "cuda" and not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("the Keras CUDA benchmark requires a TensorFlow GPU")
        with tf.device(target):
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(16,)),
                    keras.layers.Dense(64, activation="tanh"),
                    keras.layers.Dense(64, activation="tanh"),
                    keras.layers.Dense(8),
                ]
            )

        def predict(x):
            with tf.device(target):
                return model(x, training=False)

        return gb.from_tf(predict), torch.ones(32, 16, device=device, requires_grad=True)

    import warp as wp

    wp.config.quiet = True
    with wp.ScopedDevice(device):
        example = Path(__file__).resolve().parents[1] / "examples" / "newton_ball_torch.py"
        scene = runpy.run_path(str(example))["BallScene"]()

    def rollout(v):
        with wp.ScopedDevice(device):
            return scene.rollout(v)

    return gb.from_warp(rollout, in_dtypes=[wp.vec3], device=device), torch.tensor(
        [[0.0, 5.0, -5.0]], device=device, requires_grad=True
    )


def measure(dfn, x, repeats, calls):
    automatic = gb.to_torch(dfn)
    recording = _eager_function(dfn, True)
    paths = {"recording": lambda: recording.apply(x)[0], "automatic": lambda: automatic(x)}
    samples = {name: [] for name in paths}
    with torch.no_grad():
        for _ in range(5):
            for call in paths.values():
                call()
        torch.testing.assert_close(paths["recording"](), paths["automatic"]())
        for i in range(repeats):
            for name in list(paths)[:: 1 if i % 2 == 0 else -1]:
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                start = time.perf_counter()
                for _ in range(calls):
                    paths[name]()
                    if x.is_cuda:
                        torch.cuda.synchronize(x.device)
                samples[name].append((time.perf_counter() - start) * 1e3 / calls)
    return {name: statistics.median(values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["torchdiffeq", "diffrax", "keras", "newton"],
        default=["torchdiffeq", "diffrax", "keras", "newton"],
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    if args.repeats < 1 or args.calls < 1:
        parser.error("--repeats and --calls must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a PyTorch CUDA device")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    versions = ["torch", "jax", "tensorflow", "keras", "scipy", "warp-lang", "newton", "torchdiffeq", "diffrax"]
    installed = []
    for name in versions:
        try:
            installed.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    lines = [
        f"# {args.device.upper()} inference benchmark\n",
        f"{platform.machine()}, Python {platform.python_version()}; "
        f"one PyTorch thread; median of {args.repeats} rounds of {args.calls} calls; ms/call.\n",
        ", ".join(installed) + ".\n",
        "| Workload | Record backward state | Automatic inference | Speedup |",
        "|---|---:|---:|---:|",
    ]
    if args.device == "cuda":
        lines.insert(2, f"{torch.cuda.get_device_name(0)}; each call synchronized.\n")
    for name in args.workloads:
        dfn, x = workload(name, args.device)
        result = measure(dfn, x, args.repeats, args.calls)
        lines.append(
            f"| {name} | {result['recording']:.3f} | {result['automatic']:.3f} | "
            f"{result['recording'] / result['automatic']:.2f}× |"
        )
        print(lines[-1], flush=True)
    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report)


if __name__ == "__main__":
    main()
