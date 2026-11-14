"""gradbridge: gradients that cross framework boundaries.

Wrap a function from one framework, use it -- and differentiate through it --
from any other::

    import gradbridge as gb

    f = gb.from_jax(jax_fn)          # JAX function -> protocol object
    torch_f = gb.to_torch(f)         # ...callable from PyTorch
    torch_f(x).sum().backward()      # gradients flow back into JAX

Sources:   from_torch, from_jax, from_tf, from_numpy, from_warp
Consumers: to_torch,   to_jax,   to_tf,   to_numpy,   to_warp

Framework adapters are imported lazily, so only the frameworks you use need to
be installed.
"""

from ._check import CompareReport, GradcheckReport, compare, gradcheck
from ._convert import HostCopyWarning
from ._core import DiffFunction, MaxOrderError, Spec, as_diff_function

__version__ = "0.2.0"

__all__ = [
    "DiffFunction",
    "Spec",
    "MaxOrderError",
    "HostCopyWarning",
    "as_diff_function",
    "from_torch",
    "from_jax",
    "from_tf",
    "from_numpy",
    "from_warp",
    "to_torch",
    "to_jax",
    "to_tf",
    "to_numpy",
    "to_warp",
    "convert",
    "gradcheck",
    "compare",
    "GradcheckReport",
    "CompareReport",
]


def from_torch(fn, **kwargs):
    """Wrap a PyTorch function. See :func:`gradbridge.torch.from_torch`."""
    from .torch import from_torch as _f

    return _f(fn, **kwargs)


def from_jax(fn, **kwargs):
    """Wrap a JAX function. See :func:`gradbridge.jax.from_jax`."""
    from .jax import from_jax as _f

    return _f(fn, **kwargs)


def from_tf(fn, **kwargs):
    """Wrap a TensorFlow function. See :func:`gradbridge.tf.from_tf`."""
    from .tf import from_tf as _f

    return _f(fn, **kwargs)


def from_numpy(fn, **kwargs):
    """Wrap a NumPy function. See :func:`gradbridge.numpy.from_numpy`."""
    from .numpy import from_numpy as _f

    return _f(fn, **kwargs)


def from_warp(fn, **kwargs):
    """Wrap a function that launches Warp kernels. See :func:`gradbridge.warp.from_warp`."""
    from .warp import from_warp as _f

    return _f(fn, **kwargs)


def to_torch(fn, **kwargs):
    """Expose a gradbridge function to PyTorch. See :func:`gradbridge.torch.to_torch`."""
    from .torch import to_torch as _f

    return _f(fn, **kwargs)


def to_jax(fn, **kwargs):
    """Expose a gradbridge function to JAX. See :func:`gradbridge.jax.to_jax`."""
    from .jax import to_jax as _f

    return _f(fn, **kwargs)


def to_tf(fn, **kwargs):
    """Expose a gradbridge function to TensorFlow. See :func:`gradbridge.tf.to_tf`."""
    from .tf import to_tf as _f

    return _f(fn, **kwargs)


def to_numpy(fn, **kwargs):
    """Expose a gradbridge function to NumPy. See :func:`gradbridge.numpy.to_numpy`."""
    from .numpy import to_numpy as _f

    return _f(fn, **kwargs)


def to_warp(fn, **kwargs):
    """Expose a gradbridge function to Warp. See :func:`gradbridge.warp.to_warp`."""
    from .warp import to_warp as _f

    return _f(fn, **kwargs)


_SOURCES = {"torch": from_torch, "jax": from_jax, "tf": from_tf, "tensorflow": from_tf, "numpy": from_numpy, "warp": from_warp}
_CONSUMERS = {"torch": to_torch, "jax": to_jax, "tf": to_tf, "tensorflow": to_tf, "numpy": to_numpy, "warp": to_warp}


def convert(fn, *, src: str, dst: str, src_options: dict = None, dst_options: dict = None):
    """One-step bridge: ``convert(jax_fn, src="jax", dst="torch")``."""
    try:
        source, consumer = _SOURCES[src], _CONSUMERS[dst]
    except KeyError as e:
        raise ValueError(f"unknown framework {e.args[0]!r}; choose from {sorted(_SOURCES)}") from None
    return consumer(source(fn, **(src_options or {})), **(dst_options or {}))
