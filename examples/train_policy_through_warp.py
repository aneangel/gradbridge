"""Train a PyTorch network through a Warp simulation, and compare with a hand-written bridge.

An MLP maps a target (distance, height) to a launch velocity. A projectile with quadratic drag
is simulated in Warp for 200 steps (one kernel launch per step), and the loss is the squared
miss distance. The same training runs three ways:

  hand-written   a torch.autograd.Function around a wp.Tape (the usual pattern, ~18 lines)
  gradbridge     gb.to_torch(gb.from_warp(simulate))                          (1 line)
  capture=True   the same line with capture=True: forward and backward replay as CUDA graphs

All three produce identical gradients and identical training curves; the output shows the
loss live, then the cost per iteration.

Requires PyTorch and Warp with a CUDA GPU.
"""

import sys
import time

import numpy as np
import torch
import warp as wp

import gradbridge as gb

wp.config.quiet = True
wp.init()

STEPS = 200
DT = wp.constant(0.01)
DRAG = wp.constant(0.05)
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
ITERS = 300


# ============================================================================= the Warp simulator
@wp.kernel
def step(
    pos_in: wp.array(dtype=wp.vec2),
    vel_in: wp.array(dtype=wp.vec2),
    pos_out: wp.array(dtype=wp.vec2),
    vel_out: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    v = vel_in[i]
    acc = wp.vec2(0.0, -9.81) - DRAG * wp.length(v) * v
    vel_out[i] = v + DT * acc
    pos_out[i] = pos_in[i] + DT * v


def simulate(v0):
    """Launch velocities (B,) vec2 -> landing positions (B,) vec2, one launch per step."""
    n = v0.shape[0]
    pos = [wp.zeros(n, dtype=wp.vec2, requires_grad=True) for _ in range(STEPS + 1)]
    vel = [v0] + [wp.zeros(n, dtype=wp.vec2, requires_grad=True) for _ in range(STEPS)]
    for t in range(STEPS):
        wp.launch(step, dim=n, inputs=[pos[t], vel[t]], outputs=[pos[t + 1], vel[t + 1]])
    return pos[-1]


# ============================================================================= without gradbridge
class HandWritten(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v0):
        ctx.tape = wp.Tape()
        ctx.v0 = wp.from_torch(v0.detach().contiguous(), dtype=wp.vec2, requires_grad=True)
        with ctx.tape:
            ctx.out = simulate(ctx.v0)
        return wp.to_torch(ctx.out).clone()

    @staticmethod
    def backward(ctx, grad_out):
        ctx.out.grad = wp.from_torch(grad_out.contiguous(), dtype=wp.vec2)
        ctx.tape.backward()
        grad = wp.to_torch(ctx.v0.grad).clone()
        ctx.tape.zero()
        ctx.tape = ctx.v0 = ctx.out = None
        return grad


# ============================================================================= with gradbridge
METHODS = {
    "hand-written": HandWritten.apply,
    "gradbridge": gb.to_torch(gb.from_warp(simulate, in_dtypes=[wp.vec2])),
    "capture=True": gb.to_torch(gb.from_warp(simulate, in_dtypes=[wp.vec2], capture=True)),
}


# ============================================================================= training
def make_policy():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2)
    ).cuda()


def targets(it):
    g = torch.Generator(device="cuda").manual_seed(1000 + it)
    distance = 4.0 + 8.0 * torch.rand(BATCH, device="cuda", generator=g)
    height = -1.0 + 3.0 * torch.rand(BATCH, device="cuda", generator=g)
    return torch.stack([distance, height], dim=1)


def loss_fn(policy, sim, target):
    v0 = policy(target / 10.0) * 5.0 + 6.0
    return ((sim(v0) - target) ** 2).sum(dim=1).mean()


def train(name, sim):
    policy = make_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    losses, times = [], []
    for it in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = loss_fn(policy, sim, targets(it))
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if it % 50 == 0 or it == ITERS - 1:
            print(f"  {name:13s} iter {it:3d}   miss {losses[-1] ** 0.5 * 100:7.2f} cm")
    return losses, 1e3 * float(np.median(times[5:]))  # skip compile and graph-capture iterations


def main():
    print(f"Batch {BATCH} targets, {STEPS} simulation steps, {ITERS} Adam iterations\n")
    print("1. Same gradients from all three (first batch, all network parameters)")
    grads = {}
    for name, sim in METHODS.items():
        policy = make_policy()
        for _ in range(3):  # the third call of capture=True is a graph replay
            policy.zero_grad()
            loss_fn(policy, sim, targets(0)).backward()
        grads[name] = torch.cat([p.grad.flatten() for p in policy.parameters()])
    for name in ("gradbridge", "capture=True"):
        print(f"  {name:13s} bit-identical to hand-written: {torch.equal(grads[name], grads['hand-written'])}")

    print("\n2. Training, live")
    results = {name: train(name, sim) for name, sim in METHODS.items()}

    print("\n3. Cost per training iteration (median)")
    base = results["hand-written"][1]
    for name, (losses, ms) in results.items():
        same = losses == results["hand-written"][0]
        print(f"  {name:13s} {ms:6.2f} ms   {base / ms:4.1f}x   identical training curve: {same}")


if __name__ == "__main__":
    main()
