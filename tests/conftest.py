"""Shared fixtures: one test function implemented in every framework.

    F(x, y) = (sin(x) * y,  [sum(x**2)])        x, y: shape (3,), float64

VJP with cotangents (c1, c2):
    dx = c1 * cos(x) * y + 2 * x * c2
    dy = c1 * sin(x)
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# PyTorch first: see the TensorFlow/Triton LLVM note in gradbridge/torch.py.
import gradbridge.torch  # noqa: F401
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
import warp as wp

import gradbridge as gb

wp.config.quiet = True


# ----------------------------------------------------------------------------- reference
def ref_values(x, y):
    return np.sin(x) * y, np.array([np.sum(x**2)])


def ref_vjp(x, y, c1, c2):
    return c1 * np.cos(x) * y + 2 * x * c2[0], c1 * np.sin(x)


def ref_hvp_x(x, y, c1, c2, v):
    """d/dx of <dx(x,y,c), v> -- the x-part of the second-order VJP w.r.t. x."""
    return (-c1 * np.sin(x) * y + 2 * c2[0]) * v


# ----------------------------------------------------------------------------- implementations
def np_fn(x, y):
    return np.sin(x) * y, np.array([np.sum(x**2)])


def np_vjp(x, y, c1, c2):
    return c1 * np.cos(x) * y + 2 * x * c2[0], c1 * np.sin(x)


def torch_fn(x, y):
    return torch.sin(x) * y, (x**2).sum().reshape(1)


def jax_fn(x, y):
    return jnp.sin(x) * y, jnp.sum(x**2).reshape(1)


def tf_fn(x, y):
    return tf.sin(x) * y, tf.reshape(tf.reduce_sum(x**2), [1])


@wp.kernel
def _warp_kernel(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    out1: wp.array(dtype=wp.float64),
    out2: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    out1[i] = wp.sin(x[i]) * y[i]
    wp.atomic_add(out2, 0, x[i] * x[i])


def warp_fn(x, y):
    out1 = wp.zeros(x.shape, dtype=wp.float64, requires_grad=True, device=x.device)
    out2 = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=x.device)
    wp.launch(_warp_kernel, dim=x.shape[0], inputs=[x, y], outputs=[out1, out2], device=x.device)
    return out1, out2


def make_source(name):
    if name == "numpy":
        return gb.from_numpy(np_fn, vjp=np_vjp, name="F_numpy")
    if name == "numpy_fd":
        return gb.from_numpy(np_fn, name="F_numpy_fd")
    if name == "torch":
        return gb.from_torch(torch_fn, name="F_torch")
    if name == "jax":
        return gb.from_jax(jax_fn, name="F_jax")
    if name == "tf":
        return gb.from_tf(tf_fn, name="F_tf")
    if name == "warp":
        return gb.from_warp(warp_fn, name="F_warp")
    raise ValueError(name)


SOURCES = ["numpy", "torch", "jax", "tf", "warp"]
SECOND_ORDER_SOURCES = ["numpy_fd", "torch", "jax", "tf"]


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    return dict(
        x=rng.standard_normal(3),
        y=rng.standard_normal(3),
        c1=rng.standard_normal(3),
        c2=rng.standard_normal(1),
        v=rng.standard_normal(3),
    )
