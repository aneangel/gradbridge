# Downstream validation

The tests in [test_downstream.py](../tests/test_downstream.py) use real downstream APIs,
with small deterministic problems whose answers can be checked independently.
They cover the five framework ecosystems through an actual downstream library in each case.
The [user guide](guide.md) describes the APIs; the [limitation audit](limitations-audit.md)
separately records native-callback and compiler regression checks.

| Ecosystem | Downstream project | What is checked |
|---|---|---|
| NumPy | [SciPy CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html) | PyTorch `jacfwd` and reverse gradients against the spline's analytic derivative, using explicit NumPy JVP/VJP rules |
| PyTorch | [torchdiffeq](https://github.com/rtqichen/torchdiffeq) | RK4 ODE solution, JAX JVP and jitted reverse gradient against exponential decay |
| JAX | [Diffrax](https://docs.kidger.site/diffrax/) | ODE solution, PyTorch JVP, `jacfwd` and backward against exponential decay; `DirectAdjoint` enables both modes |
| TensorFlow | [Keras](https://keras.io/api/layers/core_layers/dense/) | Two-layer model's PyTorch JVP, full forward Jacobian and backward against native TensorFlow `ForwardAccumulator` and `GradientTape.jacobian` |
| Warp | [Newton](https://github.com/newton-physics/newton) | 16-step particle simulation through `SolverSemiImplicit`; PyTorch backward against a native Warp tape and the analytic velocity sensitivity |

Every case also verifies that a PyTorch call under `no_grad()` does not invoke the
source's forward-with-residuals path, including when its input requires gradients. These
are focused integration checks, not claims of compatibility with every feature of a project.
Newton remains first-order and has no forward-mode rule.

Install the optional test dependencies and run:

```bash
pip install -e ".[all,test,downstream]"
pytest tests/test_downstream.py tests/test_torch_forward.py -q -W error::gradbridge.HostCopyWarning
```

CPU CI is configured to install the downstream extra and run these tests. The downstream and
PyTorch forward-mode tests include GPU variants that skip when the required frameworks have no GPU.

## Tested environment

The recorded validation and newer performance measurements used this environment on
2026-11-13. This is one tested configuration, not a full version-compatibility matrix.

| Component | Version or hardware |
|---|---|
| Platform | Linux, Python 3.12.13 |
| CPU | AMD Ryzen 7 7800X3D |
| GPU | NVIDIA GeForce RTX 5090, 32 GB; driver 615.71.10 |
| PyTorch | 2.10.0+cu128 |
| JAX | 0.12.0.1; matching `jax-cuda12-plugin` and `jax-cuda12-pjrt` |
| TensorFlow / Keras | 2.21.0 / 3.15.1 |
| Warp / Newton | 1.16.0 / 1.6.0.dev0 |
| SciPy | 1.18.1 |
| torchdiffeq / Diffrax | 0.2.5 / 0.7.2 |

The [framework benchmark matrix](../benchmarks/RESULTS.md#framework-callback-matrix)
uses the current code and the framework versions above.

## GPU validation

CUDA execution was verified in PyTorch, JAX, TensorFlow and Warp on the RTX 5090 in the
environment above. All five downstream integrations passed on both CPU and CUDA.

The complete CPU/GPU suite passed: **505 passed, zero skipped**, with
`-W error::gradbridge.HostCopyWarning`. This includes all CUDA graph, forward-mode and
downstream tests. The only warning was the expected finite-difference fallback in the
NumPy second-order gradient check.

The [limitation audit](limitations-audit.md) adds 71 regression cases in
[test_limitation_fixes.py](../tests/test_limitation_fixes.py) for compiled forward
mode, native scalar/high-rank/batched callbacks, compiler-free GPU buffer sharing, explicit
JVP rules, mutation notification, CUDA graph array controls, placement and native TensorFlow XLA export.

The downstream tests now run on both CPU and CUDA, check output and derivative placement,
and compare against the same native or analytic references. NumPy/SciPy computation remains
on the host by design; its CUDA case checks that PyTorch inputs and returned derivatives
are on the GPU. GPU-enabled differentiable frameworks compute on the device.

The test process uses `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`TF_FORCE_GPU_ALLOW_GROWTH=true` so the four frameworks can share the GPU. This environment
also has JAX's matching `jax-cuda12-plugin` and `jax-cuda12-pjrt` packages, version 0.12.0.1.
The install extra above does not by itself select CUDA-enabled wheels for every framework.
Reproduce with GPU access, CUDA-enabled framework installations and the matching JAX GPU
plugins already provisioned:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true \
    pytest -q -W error::gradbridge.HostCopyWarning
```

## Inference performance

Eager PyTorch chooses evaluation without residuals under `no_grad()` or when no tensor
input requires gradients. Training with gradients enabled retains the residual-caching path.

[bench_inference.py](../benchmarks/bench_inference.py) compares automatic inference with
explicit residual recording. Both paths run under `no_grad()`
on identical inputs; their outputs are checked before timing. They are warmed up, then
measured in alternating order to reduce ordering bias. These measurements cover inference
only; they do not measure a training speedup.

CPU measurements, AMD Ryzen 7 7800X3D, Python 3.12.13, one PyTorch and TensorFlow thread, median of seven rounds
of 20 calls (milliseconds per call):

| Workload | Record backward state | Automatic inference | Speedup |
|---|---:|---:|---:|
| torchdiffeq: RK4, 20 steps, 16 states | 1.499 | 1.170 | 1.28× |
| Diffrax: Tsit5, 20 steps, 16 states | 0.103 | 0.082 | 1.26× |
| Keras: 16→64→64→8 MLP, batch 32 | 1.229 | 1.131 | 1.09× |
| Newton: contact rollout from `newton_ball_torch.py`, 288 substeps | 34.095 | 31.066 | 1.10× |

Versions are recorded in the [tested environment](#tested-environment) above. The Keras
integration test uses two layers; this inference benchmark uses three dense layers.

Reproduce from the repository root:

```bash
python benchmarks/bench_inference.py --markdown inference.md
```

GPU measurements on the RTX 5090, using the same versions, inputs, warmup and timing rounds.
Each CUDA call is synchronized before stopping the clock (milliseconds per call):

| Workload | Record backward state | Automatic inference | Speedup |
|---|---:|---:|---:|
| torchdiffeq: RK4, 20 steps, 16 states | 3.545 | 3.173 | 1.12× |
| Diffrax: Tsit5, 20 steps, 16 states | 3.934 | 3.651 | 1.08× |
| Keras: 16→64→64→8 MLP, batch 32 | 1.397 | 1.302 | 1.07× |
| Newton: contact rollout, 288 substeps | 29.777 | 24.861 | 1.20× |

```bash
python benchmarks/bench_inference.py --device cuda --markdown inference-gpu.md
```

The torchdiffeq benchmark sets `from_torch(..., device=...)`, which exercises the bridge
and its source residuals; without explicit placement, PyTorch-to-PyTorch calls compose
natively and already avoid bridge overhead. The Newton benchmark reuses the existing
contact simulation example on the selected device; the integration test uses a smaller free-flight scene
with an analytic derivative. Timings depend on hardware, framework versions and thread settings.

## Other optimization measurements

The downstream timings above measure inference only. Two additional scripts measure
different changes, with checked values and gradients and synchronized GPU timings:

| Workload | Baseline | Recorded RTX 5090 speedup |
|---|---|---:|
| PyTorch source called from JAX: scalar, batched and five-dimensional signatures | Forced host callback using the same current derivative rules | 5.69× / 19.61× / 40.48× |
| Exported JAX function and surrounding TensorFlow math, value and gradient | Device callback in `tf.function`, compared with native graph + whole-function XLA | 5.68× |

These are focused one-million-element benchmarks, not downstream model training speedups.
See [methodology, raw timings and commands](limitations-audit.md#measured-effect-of-removing-host-fallbacks),
or run the scripts from the repository root:

```bash
python benchmarks/bench_jax_native.py --markdown native-callbacks.md
python benchmarks/bench_tf_native.py --markdown native-tf.md --hlo optimized.hlo
```

Current [simulation and policy training measurements](../benchmarks/RESULTS.md#warp-simulation-and-policy-workloads)
also compare native Warp, hand-written bridges and gradbridge with/without CUDA graphs.
Newton's six paths have bit-identical input gradients; the projectile's four paths have
identical checked parameter gradients and 300-step training curves.
