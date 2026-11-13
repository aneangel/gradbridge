"""Array conversion between frameworks.

Memory is shared through DLPack when both sides support it (zero-copy), and
falls back to a host copy through NumPy otherwise. Frameworks are imported
lazily so gradbridge only needs the ones you actually use.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np


class HostCopyWarning(UserWarning):
    """Emitted when an array that could be shared zero-copy is copied through host memory instead."""


def _warn_host_copy(src: str, dst: str, err: Exception) -> None:
    warnings.warn(
        f"gradbridge: DLPack export {src} -> {dst} failed ({type(err).__name__}: {err}); copying through host memory.",
        HostCopyWarning,
        stacklevel=3,
    )

_FRAMEWORK_BY_MODULE = {
    "torch": "torch",
    "jax": "jax",
    "jaxlib": "jax",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "warp": "warp",
    "numpy": "numpy",
}


def framework_of(x) -> str:
    """Return the framework name that owns array ``x``."""
    mod = type(x).__module__.split(".")[0]
    return _FRAMEWORK_BY_MODULE.get(mod, "numpy")


# ----------------------------------------------------------------------------- numpy
def to_numpy(x) -> np.ndarray:
    fw = framework_of(x)
    if fw == "torch":
        return x.detach().resolve_conj().resolve_neg().cpu().numpy()
    if fw == "jax":
        a = np.asarray(x)
        if a.dtype.names is not None or str(a.dtype) == "[('float0', 'V')]":  # jax float0
            return np.zeros(a.shape, dtype=np.float32)
        return a
    if fw == "tensorflow":
        return x.numpy()
    if fw == "warp":
        return x.numpy()
    return np.asarray(x)


def wp_to_torch(a):
    """Zero-copy Warp CUDA array -> torch tensor, ordered after Warp's pending work on the stream."""
    import warp as wp

    import torch

    _torch_after_warp(torch.device(f"cuda:{a.device.ordinal}"))
    t = wp.to_torch(a, requires_grad=False)
    t._warp_array = a  # keep the Warp allocation alive as long as the tensor
    return t


def numpy_dtype(x) -> np.dtype:
    return to_numpy(x).dtype


def is_float(x) -> bool:
    """Whether ``x`` holds floating/complex values, without copying it off the device."""
    fw = framework_of(x)
    if fw == "torch":
        return x.is_floating_point() or x.is_complex()
    if fw == "tensorflow":
        return x.dtype.is_floating or x.dtype.is_complex
    if fw == "warp":
        import warp as wp

        # Vector/matrix/quaternion types carry their component type in _wp_scalar_type_.
        return wp.types.type_is_float(getattr(x.dtype, "_wp_scalar_type_", x.dtype))
    return np.issubdtype(np.dtype(np.asarray(x).dtype if fw == "numpy" else x.dtype), np.inexact)


def _is_float0(x) -> bool:
    """JAX's float0: the cotangent type of integer inputs, which has no DLPack representation."""
    return framework_of(x) == "jax" and str(x.dtype) == "float0"


def _dlpack_ok(x) -> bool:
    if not hasattr(x, "__dlpack__"):
        return False
    if isinstance(x, np.ndarray) and (not x.flags.writeable or not x.flags.c_contiguous):
        return False
    return True


# ----------------------------------------------------------------------------- torch <-> warp streams
# Torch <-> Warp conversions order the two frameworks' CUDA streams with one event wait instead
# of a DLPack capsule handshake per tensor (several times cheaper; same ordering guarantee).
_TORCH_STREAM_WRAPPERS: dict = {}


def _torch_and_warp_streams(torch_device):
    """(Warp's current stream, a Warp wrapper of torch's current stream) on ``torch_device``."""
    import torch
    import warp as wp

    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    wp_stream = wp.get_stream(f"cuda:{index}")
    torch_stream = torch.cuda.current_stream(index)
    key = (index, torch_stream.cuda_stream)
    wrapper = _TORCH_STREAM_WRAPPERS.get(key)
    if wrapper is None:
        wrapper = _TORCH_STREAM_WRAPPERS[key] = wp.stream_from_torch(torch_stream)
    return wp_stream, wrapper


def _warp_after_torch(torch_device):
    wp_stream, torch_stream = _torch_and_warp_streams(torch_device)
    if wp_stream.cuda_stream != torch_stream.cuda_stream:
        wp_stream.wait_stream(torch_stream)


def _torch_after_warp(torch_device):
    wp_stream, torch_stream = _torch_and_warp_streams(torch_device)
    if wp_stream.cuda_stream != torch_stream.cuda_stream:
        torch_stream.wait_stream(wp_stream)


# ----------------------------------------------------------------------------- torch
def to_torch(x, device=None, dtype=None):
    import torch

    if framework_of(x) == "torch":
        t = x
    elif framework_of(x) == "warp":
        if x.device.is_cuda:
            # wp.to_torch aliases without ordering the streams; order them explicitly so torch
            # never reads an output a Warp kernel is still writing.
            t = wp_to_torch(x)
        else:
            t = torch.from_dlpack(x)
    else:
        t = None
        if framework_of(x) == "tensorflow":
            # TensorFlow rejects the stream argument of __dlpack__, but its to_dlpack blocks
            # until the tensor's GPU work has finished, so torch then reads final memory.
            import tensorflow as tf

            try:
                t = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(x))
            except Exception as e:  # pragma: no cover - depends on dtype/device
                _warn_host_copy("tensorflow", "torch", e)
        elif _dlpack_ok(x):
            try:
                t = torch.from_dlpack(x)
            except Exception as e:  # pragma: no cover - depends on producer
                _warn_host_copy(framework_of(x), "torch", e)
        if t is None:
            t = torch.as_tensor(np.ascontiguousarray(to_numpy(x)))
    if device is not None and t.device != torch.device(device):
        t = t.to(device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t


def torch_dtype(np_dtype):
    import torch

    return {
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("uint8"): torch.uint8,
        np.dtype("bool"): torch.bool,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
    }[np.dtype(np_dtype)]


def numpy_dtype_from_torch(t_dtype) -> np.dtype:
    """Static mapping -- must not allocate tensors (it runs under fake-tensor tracing)."""
    import torch

    table = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.bfloat16: "float32",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
        torch.complex64: "complex64",
        torch.complex128: "complex128",
    }
    return np.dtype(table[t_dtype])


# ----------------------------------------------------------------------------- jax
# Views such as x[1:] or x.select(0, b) start mid-allocation. XLA rejects buffers not aligned
# to 16 bytes at execution time; TensorFlow's Eigen kernels abort the process on anything not
# aligned to 64. Such views are copied (on their own device) before a zero-copy export.
_XLA_ALIGNMENT = 16
_TF_ALIGNMENT = 64


def _aligned(x, alignment):
    fw = framework_of(x)
    if fw == "torch" and x.data_ptr() % alignment:
        import torch

        return x.clone(memory_format=torch.contiguous_format)
    if fw == "warp" and x.ptr is not None and x.ptr % alignment:
        import warp as wp

        return wp.clone(x)
    return x


def to_jax(x, dtype=None):
    import jax
    import jax.numpy as jnp

    if framework_of(x) == "jax":
        a = x
    else:
        a = None
        if framework_of(x) == "torch":
            # PyTorch refuses to export tensors that require grad. gradbridge does its own
            # differentiation, so the autograd metadata is not needed on the JAX side.
            x = x.detach()
            # JAX imports only compact layouts; e.g. the cotangent of .sum() is an expanded
            # (stride-0) tensor. A device-side copy is far cheaper than a host round trip.
            if not x.is_contiguous():
                x = x.contiguous()
        x = _aligned(x, _XLA_ALIGNMENT)
        if framework_of(x) != "numpy" and _dlpack_ok(x):
            try:
                a = jax.dlpack.from_dlpack(x)
            except Exception as e:  # pragma: no cover - depends on producer
                _warn_host_copy(framework_of(x), "jax", e)
        if a is None:
            a = jnp.asarray(to_numpy(x))
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype)
    return a


# ----------------------------------------------------------------------------- tensorflow
@functools.lru_cache(maxsize=None)
def _tf_has_gpu() -> bool:
    """Checked up front: a capsule TensorFlow fails to import can abort the process when its
    producer's deleter runs (observed with Warp capsules and a CPU-only TensorFlow build)."""
    import tensorflow as tf

    return bool(tf.config.list_physical_devices("GPU"))


def _on_cuda(x) -> bool:
    fw = framework_of(x)
    if fw == "torch":
        return x.is_cuda
    if fw == "jax":
        return all(d.platform == "gpu" for d in x.devices())
    if fw == "warp":
        return x.device.is_cuda
    return False


def _finished_capsule(x):
    """A DLPack capsule for ``x`` whose producing GPU work has completed.

    TensorFlow imports capsules without a stream handshake, so the producer is synchronized
    first: a host wait, but no copy.
    """
    fw = framework_of(x)
    if fw == "torch":
        import torch

        x = x.detach()
        if not x.is_contiguous():
            x = x.contiguous()
        x = _aligned(x, _TF_ALIGNMENT)
        if x.is_cuda:
            torch.cuda.current_stream(x.device).synchronize()
        return torch.utils.dlpack.to_dlpack(x)
    if fw == "jax":
        return x.block_until_ready().__dlpack__()
    import warp as wp

    x = _aligned(x, _TF_ALIGNMENT)
    if x.device.is_cuda:
        wp.synchronize_device(x.device)
    return wp.to_dlpack(x)


def to_tf(x, dtype=None):
    import tensorflow as tf

    t = None
    if framework_of(x) == "tensorflow":
        t = x
    elif _on_cuda(x) and not _is_float0(x) and _tf_has_gpu():
        # Only GPU data is wrapped: TensorFlow runs eager ops where their inputs live, so
        # wrapping host data would move the TF computation to the CPU. Host data goes through
        # convert_to_tensor below, which places it on TF's default device as before.
        try:
            t = tf.experimental.dlpack.from_dlpack(_finished_capsule(x))
        except Exception as e:  # pragma: no cover - depends on dtype/alignment
            _warn_host_copy(framework_of(x), "tensorflow", e)
    if t is None:
        t = tf.convert_to_tensor(to_numpy(x))
    if dtype is not None and t.dtype != dtype:
        t = tf.cast(t, dtype)
    return t


# ----------------------------------------------------------------------------- warp
def warp_scalar_dtype(np_dtype):
    import warp as wp

    return {
        np.dtype("float16"): wp.float16,
        np.dtype("float32"): wp.float32,
        np.dtype("float64"): wp.float64,
        np.dtype("int32"): wp.int32,
        np.dtype("int64"): wp.int64,
        np.dtype("int8"): wp.int8,
        np.dtype("uint8"): wp.uint8,
        np.dtype("bool"): wp.bool,
    }[np.dtype(np_dtype)]


def to_warp(x, dtype=None, requires_grad=False, device=None):
    """Convert to a Warp array. ``dtype`` may be a Warp vector/matrix type (e.g. wp.vec3).

    Torch, JAX, Warp and GPU-resident TensorFlow arrays are wrapped zero-copy over DLPack,
    ordered after the producer's GPU work. With ``requires_grad`` the result gets a fresh gradient buffer owned
    by the result, never the source's own ``.grad``, so a Warp tape cannot write into it.
    """
    import warp as wp

    fw = framework_of(x)
    if fw == "warp" and not requires_grad and (dtype is None or x.dtype == dtype):
        return x
    if fw == "torch" and x.is_cuda:
        src = x.detach()
        if not src.is_contiguous():
            src = src.contiguous()
        _warp_after_torch(src.device)
        # requires_grad=False: never wrap the tensor's own .grad (see the docstring).
        a = wp.from_torch(src, dtype=dtype, requires_grad=False)
        a = a.to(wp.get_device(device))
        if requires_grad:
            a.requires_grad = True
        return a
    tf_on_gpu = fw == "tensorflow" and "GPU" in x.device
    if (fw in ("torch", "jax", "warp") or tf_on_gpu) and not _is_float0(x):
        src = x
        if fw == "torch":
            src = x.detach()
            if not src.is_contiguous():
                src = src.contiguous()
        elif fw == "tensorflow":
            import tensorflow as tf

            # TensorFlow rejects __dlpack__'s stream argument; to_dlpack instead blocks until
            # the tensor's GPU work has finished.
            src = tf.experimental.dlpack.to_dlpack(x)
        try:
            a = wp.from_dlpack(src, dtype=dtype)
        except Exception as e:  # pragma: no cover - depends on producer
            _warn_host_copy(fw, "warp", e)
        else:
            # Warp code commonly allocates on the default device, so inputs must land there
            # too; this is a no-op (still zero-copy) when the source already lives there.
            a = a.to(wp.get_device(device))
            if requires_grad:
                a.requires_grad = True
            return a
    a = np.ascontiguousarray(to_numpy(x))
    wdtype = dtype if dtype is not None else warp_scalar_dtype(a.dtype)
    return wp.array(a, dtype=wdtype, requires_grad=requires_grad, device=device)
