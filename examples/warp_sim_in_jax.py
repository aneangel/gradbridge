"""Optimize a Warp physics simulation with JAX gradients.

A projectile with quadratic drag is simulated in Warp (forward Euler, 200 steps).
gradbridge exposes the Warp program to JAX, and we use jax.grad + jax.jit to
find the launch velocity that lands the projectile on a target.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

import gradbridge as gb

jax.config.update("jax_enable_x64", True)
wp.config.quiet = True

STEPS, DT, DRAG = 200, 0.01, 0.05
TARGET = np.array([12.0, 0.0])


@wp.kernel
def step(
    pos_in: wp.array(dtype=wp.vec2d),
    vel_in: wp.array(dtype=wp.vec2d),
    pos_out: wp.array(dtype=wp.vec2d),
    vel_out: wp.array(dtype=wp.vec2d),
):
    i = wp.tid()
    v = vel_in[i]
    speed = wp.length(v)
    acc = wp.vec2d(wp.float64(0.0), wp.float64(-9.81)) - wp.float64(DRAG) * speed * v
    vel_out[i] = v + wp.float64(DT) * acc
    pos_out[i] = pos_in[i] + wp.float64(DT) * v


def simulate(v0):
    """Warp program: initial velocity (1, 2) -> final position (1, 2)."""
    pos = [wp.zeros(1, dtype=wp.vec2d, requires_grad=True) for _ in range(STEPS + 1)]
    vel = [v0] + [wp.zeros(1, dtype=wp.vec2d, requires_grad=True) for _ in range(STEPS)]
    for t in range(STEPS):
        wp.launch(step, dim=1, inputs=[pos[t], vel[t]], outputs=[pos[t + 1], vel[t + 1]])
    return pos[-1]


sim = gb.to_jax(gb.from_warp(simulate, in_dtypes=[wp.vec2d]))


@jax.jit
def loss(v0):
    final = sim(v0.reshape(1, 2))
    return jnp.sum((final[0] - TARGET) ** 2)


grad = jax.jit(jax.grad(loss))

v = jnp.array([8.0, 6.0])
print(f"start: v0={np.round(np.asarray(v), 3)}  loss={float(loss(v)):.4f}")
for it in range(60):
    v = v - 0.05 * grad(v)
    if it % 15 == 14:
        print(f"iter {it + 1:3d}: v0={np.round(np.asarray(v), 3)}  loss={float(loss(v)):.6f}")

final = np.asarray(sim(v.reshape(1, 2)))[0]
print(f"lands at {np.round(final, 4)}  (target {TARGET})")
assert float(loss(v)) < 1e-3, "optimization did not converge"
print("gradcheck:", gb.gradcheck(gb.from_warp(simulate, in_dtypes=[wp.vec2d]), np.array([[8.0, 6.0]])).passed)
