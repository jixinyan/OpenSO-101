# Sim-to-Real Robustness

Sim-to-real (s2r) is the gap between "the policy works in Isaac Lab" and
"the policy works on the table in front of you." Closing that gap is a
first-class concern in OpenSO-101 and is organized into two phases.

## Phase 1: Physics Domain Randomization (Today)

Phase 1 perturbs the physical parameters of the simulated SO-101 and its
environment so the trained policy sees a wide distribution of dynamics.
This is **fully wired up** in `openso101/sim2real/domain_randomization/`
and is on by default for all built-in RL tasks.

What gets randomized at each episode reset:

| Parameter | Distribution | Why |
|---|---|---|
| Gravity vector | small perturbation around (0, 0, −9.81) | Mimics table tilt |
| Cube friction | uniform [0.4, 1.2] | Real cubes vary |
| Robot joint friction / damping | log-uniform around URDF defaults | Wear and tear |
| Cube spawn pose | uniform within a small box | Real reach is imperfect |
| Object mass | uniform around the nominal | Manufacturing tolerance |

To **disable** randomization (e.g. for evaluation or video capture), use
the `play` variant:

```python
gym.make("OpenSO101-PickPlace-v0", play=True)
```

…or `openso101 rl play --task ...` which sets `play=True` automatically.

## Phase 1: Observation / Action Noise

In addition to physics perturbations, the framework supports per-step
sensor and actuator noise:

- **Observation noise** — Gaussian on joint positions, end-effector pose;
  uniform pixel noise on cameras.
- **Action noise** — Gaussian on commanded joint targets.

Configured in `openso101/sim2real/noise/`. Adjust per-task by overriding
`OpenSO101EnvCfg.events` and `OpenSO101EnvCfg.actions` in your `__post_init__`.

## Phase 2: Deployment Bridge (Future)

Sub-project B will close the loop:

- **ROS bridge** — speak the SO-101's protocol via `lerobot` device drivers,
  publish sim-equivalent observations at the same rate.
- **Real cube + table setup** — calibration scripts to match the sim's
  origin to the physical workspace.
- **Latency injection** — replay real network/USB latencies during training
  so the policy isn't surprised by them at deploy.
- **Online safety layer** — joint-limit / velocity-limit / collision-cone
  guards that wrap the policy at deploy time.

Today this lives behind:

```bash
openso101 sim2real deploy ...   # exits 2 with "future"
```

## How Phases Compose

For research transferring SO-101 sim policies to hardware, the recommended
trajectory is:

1. Train PPO with phase-1 DR on (e.g.) `OpenSO101-PickPlace-v0`.
2. Use `openso101 rl play` and a real-time playback rig to evaluate the
   policy in sim under varied seeds.
3. (Phase 2) Use the deploy bridge to run the same policy on hardware
   with the safety layer engaged.

The closer phase-1's distribution is to your physical setup, the smaller
the residual gap phase-2 has to close.

## See Also

- [Isaac Lab Manager events][isaaclab-events] — the underlying perturbation
  hook system.
- [Sim-to-real survey][s2r-survey] — for the broader research landscape.

[isaaclab-events]: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html
[s2r-survey]: https://arxiv.org/abs/2009.13303
