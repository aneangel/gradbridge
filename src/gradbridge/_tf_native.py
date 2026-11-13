"""Device-resident TensorFlow consumer: builds and drives the ``GradbridgeCall`` op (``_tf_op.cc``).

``tf.py_function`` is a host op, so TensorFlow copies GPU inputs to the CPU and results back.
``GradbridgeCall`` has CPU and GPU kernels that hand TensorFlow's own buffers and CUDA stream to
a Python callback, which runs the bridged function on them zero-copy (through Warp arrays, the
same way the JAX consumer's XLA FFI path does).

The op is compiled on first use against the installed TensorFlow (C++ compiler required, ``CXX``
overrides ``g++``) and cached per TensorFlow version under ``~/.cache/gradbridge`` (override with
``GRADBRIDGE_CACHE_DIR``). If it cannot be built or loaded, :func:`load` returns ``None`` and the
TensorFlow consumer keeps using ``tf.py_function``.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import itertools
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np

from . import _convert

_LOCK = threading.Lock()
_STATE: dict = {"tried": False, "module": None, "keepalive": None}
_REGISTRY: dict = {}  # callback id -> DiffFunction
_ERRORS: dict = {}  # thread id -> exception raised by the bridged function
_IDS = itertools.count(1)
_STREAMS: dict = {}  # (ordinal, cuda stream handle) -> wp.Stream wrapper

# TensorFlow DataType enum -> NumPy dtype, for the dtypes Warp arrays can hold.
_TF_DTYPES = {1: "float32", 2: "float64", 3: "int32", 4: "uint8", 6: "int8", 9: "int64", 10: "bool", 19: "float16"}

_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.c_int,
)


def _build() -> Path:
    import tensorflow as tf

    src = Path(__file__).with_name("_tf_op.cc")
    flags = tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    key = hashlib.sha256((src.read_text() + tf.__version__ + " ".join(flags)).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("GRADBRIDGE_CACHE_DIR", Path.home() / ".cache" / "gradbridge"))
    out = cache / f"tf-{tf.__version__}-{key}" / "gradbridge_tf.so"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    cmd = [os.environ.get("CXX", "g++"), "-O2", "-shared", "-fPIC", str(src), "-o", str(tmp), *flags]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compiling {src.name} failed:\n{result.stderr[-2000:]}")
    os.replace(tmp, out)  # atomic: concurrent first uses never load a half-written library
    return out


def load():
    """The compiled op module, or ``None`` when it cannot be built here (callers fall back)."""
    if _STATE["tried"]:
        return _STATE["module"]
    with _LOCK:
        if _STATE["tried"]:
            return _STATE["module"]
        try:
            import tensorflow as tf

            path = _build()
            module = tf.load_op_library(str(path))
            lib = ctypes.CDLL(str(path))
            callback = _CALLBACK(_callback)
            lib.gradbridge_tf_set_callback(callback)
            _STATE["module"], _STATE["keepalive"] = module, (lib, callback)
        except Exception as e:  # pragma: no cover - depends on the toolchain
            warnings.warn(
                "gradbridge: could not build the TensorFlow consumer op, so calls from TensorFlow copy "
                f"through host memory ({type(e).__name__}: {str(e)[:300]})",
                _convert.HostCopyWarning,
                stacklevel=3,
            )
        _STATE["tried"] = True
    return _STATE["module"]


def callback_id(dfn) -> int:
    cache = dfn._consumer_cache
    if "tf_native_id" not in cache:
        cid = next(_IDS)
        _REGISTRY[cid] = dfn
        cache["tf_native_id"] = cid
    return cache["tf_native_id"]


def supported(specs) -> bool:
    """Warp arrays need 1-4 dimensions and a Warp scalar dtype; anything else uses py_function."""
    for shape, dtype in specs:
        if not 1 <= len(shape) <= 4 or np.dtype(dtype).name not in _TF_DTYPES.values():
            return False
    return True


def raise_pending():
    """Re-raise the bridged function's own exception (instead of TensorFlow's InternalError)."""
    exc = _ERRORS.pop(threading.get_ident(), None)
    if exc is not None:
        raise exc


def _stream(wp, device, handle):
    key = (device.ordinal, handle)
    s = _STREAMS.get(key)
    if s is None:
        s = _STREAMS[key] = wp.Stream(device, cuda_stream=handle)
    return s


def _callback(cid, n_in, in_ptrs, in_dims, in_ranks, in_dtypes, n_out, out_ptrs, out_dims, out_ranks, out_dtypes, stream, ordinal):
    try:
        import warp as wp

        dfn = _REGISTRY[cid]
        device = wp.get_device(f"cuda:{ordinal}") if ordinal >= 0 else wp.get_device("cpu")

        def arrays(n, ptrs, dims, ranks, dtypes):
            result, k = [], 0
            for i in range(n):
                shape = tuple(dims[k + j] for j in range(ranks[i]))
                k += ranks[i]
                dtype = _convert.warp_scalar_dtype(np.dtype(_TF_DTYPES[dtypes[i]]))
                result.append(wp.array(ptr=ptrs[i] or 0, dtype=dtype, shape=shape, device=device, copy=False))
            return result

        ins = arrays(n_in, in_ptrs, in_dims, in_ranks, in_dtypes)
        outs = arrays(n_out, out_ptrs, out_dims, out_ranks, out_dtypes)
        scope = wp.ScopedStream(_stream(wp, device, stream), sync_enter=False) if stream else contextlib.nullcontext()
        with scope:
            # TensorFlow's stream is Warp's current stream here: the sources' conversions order
            # their own streams against it in both directions, and the copies below run on it.
            results = dfn.eval(*ins)
            for dst, res in zip(outs, results):
                src = _convert.to_warp(res, device=dst.device)
                if src.dtype != dst.dtype:
                    src = wp.array(np.asarray(_convert.to_numpy(res), dtype=np.dtype(dst.dtype._type_)), device=dst.device)
                wp.copy(dst, src.reshape(dst.shape) if src.shape != dst.shape else src)
            if stream:
                # Results may be freed (and reused by another allocator) once this returns.
                wp.synchronize_stream(wp.get_stream(device))
        return 0
    except BaseException as e:  # noqa: BLE001 - must not propagate into C++
        _ERRORS[threading.get_ident()] = e
        print(f"gradbridge: {type(e).__name__} in bridged function: {e}", file=sys.stderr)
        return 1
