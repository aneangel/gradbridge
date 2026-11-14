"""Device-resident TensorFlow consumer: builds and drives the ``GradbridgeCall`` op (``_tf_op.cc``).

``GradbridgeCall`` has CPU and GPU kernels that hand TensorFlow's own buffers and CUDA stream to the
shared native bridge (``_native.py``). If the op cannot be built or loaded, :func:`load` returns
``None`` and the TensorFlow consumer uses a DLPack-aware ``tf.py_function`` callback instead.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from ._native import NativeBridge, build

# TensorFlow DataType enum -> NumPy dtype, for the dtypes Warp arrays can hold.
_TF_DTYPES = {1: "float32", 2: "float64", 3: "int32", 4: "uint8", 6: "int8", 9: "int64", 10: "bool", 19: "float16"}

_BRIDGE = NativeBridge("TensorFlow consumer", _TF_DTYPES, fallback_warning=RuntimeWarning)


def _loader():
    import tensorflow as tf

    flags = tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()
    path = build(Path(__file__).with_name("_tf_op.cc"), "tf", tf.__version__, flags)
    module = tf.load_op_library(str(path))
    lib = ctypes.CDLL(str(path))
    lib.gradbridge_tf_set_callback(_BRIDGE.callback)
    return module, lib


def load():
    """The compiled op module, or ``None`` when it cannot be built here (callers fall back)."""
    return _BRIDGE.load(_loader)


callback_id = _BRIDGE.callback_id
supported = _BRIDGE.supported
raise_pending = _BRIDGE.raise_pending
