"""Train through a Newton physics simulation from PyTorch.

The scene is Newton's own ``example_diffsim_ball``: a particle thrown at a wall, bouncing off it
and the ground (soft contacts, ``SolverSemiImplicit``, 36 frames x 8 substeps). gradbridge exposes
the Warp rollout to PyTorch, so PyTorch optimizers and networks train through the contacts.

1. Equivalence: ``torch.optim.SGD`` through gradbridge reproduces Newton's native training loop.
2. A PyTorch MLP policy learns ``target -> launch velocity`` through the simulation.
3. Cost per iteration, each approach without and with CUDA graphs: Newton's native loop (with
   graphs exactly as ``example_diffsim_ball`` captures them), a hand-written
   ``torch.autograd.Function`` (with graphs written by hand), and gradbridge (``capture=True``).
   Every approach is checked to produce the same gradient.

Requires ``newton`` (https://github.com/newton-physics/newton) and a CUDA GPU.
"""

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True  # before importing newton, which initializes Warp

import newton  # noqa: E402

import gradbridge as gb  # noqa: E402

FPS, FRAMES, SUBSTEPS = 60, 36, 8
DT = 1.0 / FPS / SUBSTEPS
START_VEL = (0.0, 5.0, -5.0)
TARGET = (0.0, -2.0, 1.5)


@wp.kernel
def _set_velocity(v0: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3)):
    qd[0] = v0[0]


@wp.kernel
def _copy_position(q: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    out[0] = q[0]


@wp.kernel
def _loss(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def _sgd(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), lr: float):
    x[0] = x[0] - grad[0] * lr


class BallScene:
    """The scene of Newton's example_diffsim_ball, with the rollout as a plain Warp function."""

    def __init__(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particle(pos=wp.vec3(0.0, -0.5, 1.0), vel=wp.vec3(*START_VEL), mass=1.0)
        ke, kf, kd, mu = 1.0e4, 0.0, 1.0e1, 0.2
        cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu)
        builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()), hx=1.0, hy=0.25, hz=1.0, cfg=cfg
        )
        builder.add_ground_plane(cfg=cfg)
        self.model = builder.finalize(requires_grad=True)
        self.model.soft_contact_ke, self.model.soft_contact_kf = ke, kf
        self.model.soft_contact_kd, self.model.soft_contact_mu = kd, mu
        self.model.soft_contact_restitution = 1.0
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.states = [self.model.state() for _ in range(FRAMES * SUBSTEPS + 1)]
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(
            self.model, broad_phase="explicit", soft_contact_gap=10.0, requires_grad=True
        )
        self.contacts = pipeline.contacts()
        pipeline.collide(self.states[0], self.contacts)  # one-shot contacts: the shapes are static

    def rollout(self, v0):
        """Launch with velocity ``v0`` (one vec3) and return the final position (one vec3)."""
        wp.launch(_set_velocity, dim=1, inputs=[v0], outputs=[self.states[0].particle_qd])
        for t in range(FRAMES * SUBSTEPS):
            self.states[t].clear_forces()
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, DT)
        # A fresh output: the state buffers are reused by the next rollout.
        out = wp.empty(1, dtype=wp.vec3, requires_grad=True)
        wp.launch(_copy_position, dim=1, inputs=[self.states[-1].particle_q], outputs=[out])
        return out


# ============================================================================= 1. equivalence
def native_training(iters, lr):
    """Newton's own loop: Warp tape, loss kernel, gradient step kernel."""
    scene = BallScene()
    v0 = wp.array([START_VEL], dtype=wp.vec3, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    history = []
    for _ in range(iters):
        tape = wp.Tape()
        with tape:
            wp.launch(_loss, dim=1, inputs=[scene.rollout(v0), wp.vec3(*TARGET)], outputs=[loss])
        tape.backward(loss)
        history.append(float(loss.numpy()[0]))
        wp.launch(_sgd, dim=1, inputs=[v0, v0.grad, lr])
        tape.zero()
    return history


def torch_training(iters, lr, capture=False):
    """The same optimization, written in PyTorch: the simulator is just a differentiable function."""
    simulate = gb.to_torch(gb.from_warp(BallScene().rollout, in_dtypes=[wp.vec3], capture=capture))
    v0 = torch.tensor([START_VEL], device="cuda", requires_grad=True)
    target = torch.tensor([TARGET], device="cuda")
    opt = torch.optim.SGD([v0], lr=lr)
    history = []
    for _ in range(iters):
        loss = ((simulate(v0) - target) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


# ============================================================================= 2. policy
def sample_targets(n, generator):
    lo = torch.tensor([-0.4, -2.4, 1.2], device="cuda")
    hi = torch.tensor([0.4, -1.6, 1.8], device="cuda")
    return lo + (hi - lo) * torch.rand(n, 3, device="cuda", generator=generator)


def train_policy(steps=120, batch=8, seed=0):
    """A PyTorch MLP maps a target position to a launch velocity, trained through Newton's contacts.

    One scene is reused for every rollout in a batch, so each rollout overwrites the state buffers
    the previous rollouts' cached tapes read. gradbridge detects that (the rollouts touch the same
    memory) and recomputes those rollouts in backward, so the gradients stay exact.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    simulate = gb.to_torch(gb.from_warp(BallScene().rollout, in_dtypes=[wp.vec3]))
    policy = torch.nn.Sequential(
        torch.nn.Linear(3, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 3)
    ).cuda()
    base = torch.tensor(START_VEL, device="cuda")
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    def miss_distance(targets):
        vels = base + policy(targets)
        finals = torch.cat([simulate(v.unsqueeze(0)) for v in vels])
        return (finals - targets).norm(dim=1)

    held_out = sample_targets(32, torch.Generator(device="cuda").manual_seed(seed + 1))
    with torch.no_grad():
        before = miss_distance(held_out).mean().item()
    for step in range(steps):
        loss = (miss_distance(sample_targets(batch, gen)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 30 == 0 or step == steps - 1:
            print(f"  step {step:3d}  batch loss {loss.item():.4f}")
    with torch.no_grad():
        after = miss_distance(held_out).mean().item()
    return before, after


# ============================================================================= 3. cost
class HandWrittenBridge(torch.autograd.Function):
    """What one writes without gradbridge: wrap the tensor for Warp, keep the tape for backward."""

    @staticmethod
    def forward(ctx, v0, scene):
        ctx.wv0 = wp.from_torch(v0.detach().contiguous(), dtype=wp.vec3, requires_grad=True)
        ctx.tape = wp.Tape()
        with ctx.tape:
            ctx.out = scene.rollout(ctx.wv0)
        return wp.to_torch(ctx.out).clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.tape.backward(grads={ctx.out: wp.from_torch(grad.contiguous(), dtype=wp.vec3)})
        g = wp.to_torch(ctx.wv0.grad).clone()
        ctx.tape.zero()
        return g, None


class RolloutGraphs:
    """CUDA graphs of one rollout, captured by hand: fixed buffers, a forward and a backward graph."""

    def __init__(self, scene):
        self.wv0 = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
        self.grad_out = wp.zeros(1, dtype=wp.vec3)
        self.tape = wp.Tape()
        with wp.ScopedCapture() as capture:
            with self.tape:
                self.out = scene.rollout(self.wv0)
        self.forward = capture.graph
        with wp.ScopedCapture() as capture:
            self.tape.backward(grads={self.out: self.grad_out})
        self.backward = capture.graph


class HandWrittenGraphedBridge(torch.autograd.Function):
    """The hand-written bridge with CUDA graphs. Correct only while every forward is followed by
    its own backward: a second forward before backward would overwrite the first's intermediates."""

    @staticmethod
    def forward(ctx, v0, graphs):
        wp.copy(graphs.wv0, wp.from_torch(v0.detach().contiguous(), dtype=wp.vec3))
        wp.capture_launch(graphs.forward)
        ctx.graphs = graphs
        return wp.to_torch(graphs.out).clone()

    @staticmethod
    def backward(ctx, grad):
        graphs = ctx.graphs
        wp.copy(graphs.grad_out, wp.from_torch(grad.contiguous(), dtype=wp.vec3))
        wp.capture_launch(graphs.backward)
        g = wp.to_torch(graphs.wv0.grad).clone()
        graphs.tape.zero()
        return g, None


def time_ms(step, reps=30):
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


def cost_per_iteration():
    """Median ms per forward + backward for each approach, and the gradient each one produced."""
    target = torch.tensor([TARGET], device="cuda")
    target_wp = wp.vec3(*TARGET)
    results = {}

    def native(graphed):
        # Newton's loop: a tape around rollout + loss, backward from the loss, tape.zero() after.
        # With graphs, forward and backward are captured as one graph, as example_diffsim_ball does.
        scene = BallScene()
        wv0 = wp.array([START_VEL], dtype=wp.vec3, requires_grad=True)
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tapes = [wp.Tape()]

        def forward_backward():
            with tapes[0]:
                wp.launch(_loss, dim=1, inputs=[scene.rollout(wv0), target_wp], outputs=[loss])
            tapes[0].backward(loss)

        graph = None
        if graphed:
            with wp.ScopedCapture() as capture:
                forward_backward()
            graph = capture.graph
        grad = []

        def step():
            if graph is not None:
                wp.capture_launch(graph)
            else:
                tapes[0] = wp.Tape()
                forward_backward()
            grad[:] = [wv0.grad.numpy().copy()] if not grad else grad
            tapes[0].zero()

        return step, grad

    def bridged(fn):
        v0 = torch.tensor([START_VEL], device="cuda", requires_grad=True)
        grad = []

        def step():
            v0.grad = None
            ((fn(v0) - target) ** 2).sum().backward()
            grad[:] = [v0.grad.cpu().numpy()] if not grad else grad

        return step, grad

    hand_scene, graphs = BallScene(), RolloutGraphs(BallScene())
    approaches = {
        "native Warp": native(graphed=False),
        "native Warp, CUDA graph": native(graphed=True),
        "hand-written autograd.Function": bridged(lambda v: HandWrittenBridge.apply(v, hand_scene)),
        "hand-written, CUDA graphs": bridged(lambda v: HandWrittenGraphedBridge.apply(v, graphs)),
        "gradbridge": bridged(gb.to_torch(gb.from_warp(BallScene().rollout, in_dtypes=[wp.vec3]))),
        "gradbridge, capture=True": bridged(
            gb.to_torch(gb.from_warp(BallScene().rollout, in_dtypes=[wp.vec3], capture=True))
        ),
    }
    for name, (step, grad) in approaches.items():
        results[name] = (time_ms(step), grad[0].reshape(-1))
    return results


def main():
    iters, lr = 30, 0.02
    print("1. Equivalence with Newton's native training loop")
    native = native_training(iters, lr)
    bridged = torch_training(iters, lr)
    graphed = torch_training(iters, lr, capture=True)
    diff = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(native * 2, bridged + graphed))
    for i in (0, 10, 20, iters - 1):
        print(f"  iter {i:2d}  native {native[i]:.6f}   torch+gradbridge {bridged[i]:.6f}   capture=True {graphed[i]:.6f}")
    print(f"  max relative difference over {iters} iterations: {diff:.2e}")

    print("\n2. PyTorch MLP policy trained through Newton (target -> launch velocity)")
    before, after = train_policy()
    print(f"  mean miss distance on 32 held-out targets: {before:.3f} m -> {after:.3f} m")

    print("\n3. Cost per optimization iteration (median ms), each without and with CUDA graphs")
    results = cost_per_iteration()
    reference = results["native Warp"][1]
    for name, (ms, grad) in results.items():
        print(f"  {name:32s} {ms:7.2f}   same gradient as native: {np.array_equal(grad, reference)}")


if __name__ == "__main__":
    main()
