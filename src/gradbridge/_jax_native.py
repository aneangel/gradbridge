"""gradbridge's own XLA FFI handler for the JAX consumer (``_jax_op.cc``).

XLA hands the handler its buffers and CUDA stream; the C++ side passes raw pointers to the shared
native bridge (``_native.py``), avoiding a Python-level parse of XLA's call frame on every call. If
the handler cannot be built or registered, :func:`load` returns ``None`` and the JAX consumer
falls back to Warp's ``jax_callable``, then to ``jax.pure_callback``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from ._native import NativeBridge, build

# XLA FFI DataType codes (xla/ffi/api/c_api.h) -> NumPy dtype, for the dtypes Warp arrays can hold.
_XLA_DTYPES = {1: "bool", 2: "int8", 4: "int32", 5: "int64", 6: "uint8", 10: "float16", 11: "float32", 12: "float64"}
TARGET = "gradbridge_call"

_BRIDGE = NativeBridge("JAX consumer", _XLA_DTYPES)


def _loader():
    import jax
    import jaxlib

    flags = [f"-I{jax.ffi.include_dir()}", "-std=c++17"]
    path = build(Path(__file__).with_name("_jax_op.cc"), "jax", jaxlib.__version__, flags)
    lib = ctypes.CDLL(str(path))
    lib.gradbridge_jax_set_callback(_BRIDGE.callback)
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(lib.GradbridgeCallCuda), platform="CUDA")
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(lib.GradbridgeCallCpu), platform="cpu")
    return TARGET, lib


def load():
    """The registered FFI target name, or ``None`` when the handler cannot be built here."""
    return _BRIDGE.load(_loader)


callback_id = _BRIDGE.callback_id
supported = _BRIDGE.supported
raise_pending = _BRIDGE.raise_pending
