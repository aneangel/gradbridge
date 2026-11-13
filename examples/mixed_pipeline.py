"""One training loop, three frameworks, one backward pass.

    policy (PyTorch)  ->  dynamics (JAX)  ->  reward (TensorFlow)  ->  loss (PyTorch)

A PyTorch policy outputs actions; a JAX function rolls a simple system forward;
a TensorFlow function scores the result. gradbridge makes the whole chain
differentiable from PyTorch, so a single loss.backward() trains the policy.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gradbridge.torch  # noqa: F401  (import before TensorFlow; see README)

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch

import gradbridge as gb

jax.config.update("jax_enable_x64", False)
torch.manual_seed(0)


# --- JAX: damped point-mass dynamics, 20 steps -------------------------------
def rollout(state, actions):
    def body(s, a):
        pos, vel = s[..., 0], s[..., 1]
        vel = 0.95 * vel + 0.1 * a
        pos = pos + 0.1 * vel
        return jnp.stack([pos, vel], axis=-1), pos

    final, _ = jax.lax.scan(body, state, actions.T)
    return final


# --- TensorFlow: reward for ending at position 1.0 with zero velocity ---------
def reward(final_state):
    pos, vel = final_state[:, 0], final_state[:, 1]
    return -tf.reduce_mean((pos - 1.0) ** 2 + 0.1 * vel**2)


dynamics = gb.to_torch(gb.from_jax(rollout))
score = gb.to_torch(gb.from_tf(reward))

# --- PyTorch: policy maps initial state -> 20 actions -------------------------
policy = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 20))
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

for it in range(150):
    s0 = torch.randn(64, 2) * 0.5
    actions = policy(s0)
    loss = -score(dynamics(s0, actions))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 50 == 0 or it == 149:
        print(f"iter {it:3d}  loss {loss.item():.4f}")

assert loss.item() < 0.05, "training did not converge"
print("trained a PyTorch policy through JAX dynamics and a TensorFlow reward")
