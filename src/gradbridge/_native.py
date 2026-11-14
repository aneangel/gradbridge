"""Shared runtime for gradbridge's compiled consumer handlers (TensorFlow op, JAX XLA FFI handler).

Each handler is a small C++ library compiled on first use against the installed framework and
cached under ``~/.cache/gradbridge`` (``GRADBRIDGE_CACHE_DIR`` overrides; ``CXX`` overrides
``g++``). The C++ side reads the framework's own buffers and CUDA stream and calls one registered
Python callback with raw pointers; :class:`NativeBridge` wraps them (as PyTorch tensors for a
PyTorch source, as Warp arrays otherwise), runs the bridged function ordered on that
stream, and copies the results into the framework's outputs.
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

CALLBACK_TYPE = ctypes.CFUNCTYPE(
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


def build(source: Path, tag: str, version: str, flags: list) -> Path:
    """Compile ``source`` into a shared library, cached by source, framework version and flags."""
    key = hashlib.sha256((source.read_text() + version + " ".join(flags)).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("GRADBRIDGE_CACHE_DIR", Path.home() / ".cache" / "gradbridge"))
    out = cache / f"{tag}-{version}-{key}" / f"gradbridge_{tag}.so"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    cmd = [os.environ.get("CXX", "g++"), "-O2", "-shared", "-fPIC", str(source), "-o", str(tmp), *flags]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compiling {source.name} failed:\n{result.stderr[-2000:]}")
    os.replace(tmp, out)  # atomic: concurrent first uses never load a half-written library
    return out


class _CudaBuffer:
    """A raw device buffer exposed through the CUDA array interface (no ownership, no stream sync)."""

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, shape: tuple, dtype: np.dtype):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": dtype.str,
            "data": (ptr, False),
            "version": 3,
            "stream": None,  # the caller already runs on the stream that produced the data
        }


class NativeBridge:
    """Registry and Python side of one compiled handler. ``dtypes`` maps the framework's dtype
    codes to NumPy dtype names."""

    def __init__(self, name: str, dtypes: dict, *, fallback_warning=_convert.HostCopyWarning):
        self.name = name
        self.dtypes = dtypes
        self.fallback_warning = fallback_warning
        self.lock = threading.Lock()
        self.tried = False
        self.handle = None  # whatever the loader returns (op module, registered target, ...)
        self.keepalive = None
        self._registry: dict = {}
        self._errors: dict = {}
        self._ids = itertools.count(1)
        self._streams: dict = {}
        self.callback = CALLBACK_TYPE(self._callback)

    # ------------------------------------------------------------------ loading
    def load(self, loader):
        """Run ``loader()`` once; on failure warn and remember that the native path is unavailable."""
        if self.tried:
            return self.handle
        with self.lock:
            if self.tried:
                return self.handle
            try:
                self.handle, self.keepalive = loader()
            except Exception as e:  # pragma: no cover - depends on the toolchain
                warnings.warn(
                    f"gradbridge: could not build the {self.name} handler, so it falls back to a slower "
                    f"path ({type(e).__name__}: {str(e)[:300]})",
                    self.fallback_warning,
                    stacklevel=4,
                )
            self.tried = True
        return self.handle

    # ------------------------------------------------------------------ calls
    def callback_id(self, dfn) -> int:
        key = f"native_id_{self.name}"
        cache = dfn._consumer_cache
        if key not in cache:
            cid = next(self._ids)
            self._registry[cid] = dfn
            cache[key] = cid
        return cache[key]

    def supported(self, specs, *, framework=None) -> bool:
        """PyTorch views support any rank; the Warp-based path needs 1-4 dimensions."""
        names = set(self.dtypes.values())
        return all(
            (framework == "torch" or 1 <= len(shape) <= 4) and np.dtype(dtype).name in names
            for shape, dtype in specs
        )

    def raise_pending(self):
        """Re-raise the bridged function's own exception (instead of the framework's error)."""
        exc = self._errors.pop(threading.get_ident(), None)
        if exc is not None:
            raise exc

    def _stream(self, wp, device, handle):
        key = (device.ordinal, handle)
        s = self._streams.get(key)
        if s is None:
            s = self._streams[key] = wp.Stream(device, cuda_stream=handle)
        return s

    def _buffers(self, n, ptrs, dims, ranks, dtypes):
        """(pointer, shape, NumPy dtype) of each buffer the C++ side described."""
        result, k = [], 0
        for i in range(n):
            shape = tuple(dims[k + j] for j in range(ranks[i]))
            k += ranks[i]
            result.append((ptrs[i] or 0, shape, np.dtype(self.dtypes[dtypes[i]])))
        return result

    def _torch_call(self, dfn, ins, outs, stream, ordinal):
        """PyTorch source on CUDA: run it on the framework's own stream, on torch views of its buffers.

        Everything is ordered on that one stream: the inputs are ready, the copies into the outputs
        follow the computation, and memory PyTorch allocates here is stream-ordered on it, so no
        cross-stream handshake and no host wait is needed.
        """
        import torch

        key = ("torch", ordinal, stream)
        ext = self._streams.get(key)
        if ext is None:
            ext = self._streams[key] = torch.cuda.ExternalStream(stream, device=f"cuda:{ordinal}")
        with torch.cuda.stream(ext):
            results = dfn.eval(*[torch.as_tensor(_CudaBuffer(*b), device=f"cuda:{ordinal}") for b in ins])
            for (ptr, shape, dtype), res in zip(outs, results):
                view = torch.as_tensor(_CudaBuffer(ptr, shape, dtype), device=f"cuda:{ordinal}")
                view.copy_(res.reshape(shape) if tuple(res.shape) != shape else res)

    def _torch_cpu_call(self, dfn, ins, outs):
        """Borrow the caller's CPU buffers as NumPy/PyTorch views, including scalars and empty arrays."""
        import torch

        def view(ptr, shape, dtype):
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            buffer = (ctypes.c_ubyte * size).from_address(ptr)
            return np.ndarray(shape, dtype=dtype, buffer=buffer)

        inputs = [torch.from_numpy(view(*b)) for b in ins]
        results = dfn.eval(*inputs)
        for b, res in zip(outs, results):
            dst = view(*b)
            np.copyto(dst, res.detach().cpu().numpy().reshape(dst.shape), casting="unsafe")

    def _callback(self, cid, n_in, in_ptrs, in_dims, in_ranks, in_dtypes, n_out, out_ptrs, out_dims, out_ranks, out_dtypes, stream, ordinal):
        try:
            dfn = self._registry[cid]
            if dfn.framework == "torch":
                ins = self._buffers(n_in, in_ptrs, in_dims, in_ranks, in_dtypes)
                outs = self._buffers(n_out, out_ptrs, out_dims, out_ranks, out_dtypes)
                if ordinal >= 0:
                    self._torch_call(dfn, ins, outs, stream, ordinal)
                else:
                    self._torch_cpu_call(dfn, ins, outs)
                return 0

            import warp as wp

            device = wp.get_device(f"cuda:{ordinal}") if ordinal >= 0 else wp.get_device("cpu")

            def arrays(n, ptrs, dims, ranks, dtypes):
                result, k = [], 0
                for i in range(n):
                    shape = tuple(dims[k + j] for j in range(ranks[i]))
                    k += ranks[i]
                    dtype = _convert.warp_scalar_dtype(np.dtype(self.dtypes[dtypes[i]]))
                    result.append(wp.array(ptr=ptrs[i] or 0, dtype=dtype, shape=shape, device=device, copy=False))
                return result

            ins = arrays(n_in, in_ptrs, in_dims, in_ranks, in_dtypes)
            outs = arrays(n_out, out_ptrs, out_dims, out_ranks, out_dtypes)
            scope = wp.ScopedStream(self._stream(wp, device, stream), sync_enter=False) if stream else contextlib.nullcontext()
            with scope:
                # The framework's stream is Warp's current stream here: the sources' conversions
                # order their own streams against it in both directions; the copies run on it.
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
            self._errors[threading.get_ident()] = e
            print(f"gradbridge: {type(e).__name__} in bridged function: {e}", file=sys.stderr)
            return 1
