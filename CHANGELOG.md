# Changelog

All notable changes to OpenSO-101 are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026

Initial public release.

### Added
- **Unified CLI** (`openso101`) with four groups: `envs`, `rl`, `il`, `sim2real`.
- **Three built-in tasks** for the SO-101 5-DoF arm: `OpenSO101-Lift-v0`,
  `OpenSO101-PickPlace-v0`, `OpenSO101-Stack-v0`, registered via a one-decorator
  extension API (`@register_task` on `OpenSO101EnvCfg`).
- **Reinforcement learning** — PPO via `rsl_rl` with a best-checkpoint runner,
  plus a teacher→student `Distillation` path.
- **Imitation learning** — leader-arm teleoperation, crash-resilient chunked
  HDF5 recording, HDF5→LeRobot dataset conversion, and ACT/Diffusion training
  through LeRobot.
- **Sim-to-real** — physics / visual / observation domain randomization shared
  across all three tasks, and a real-arm deploy bridge driving the Feetech
  follower via LeRobot's `SO101Follower`.
- **Evaluation & observability** — `rl eval` / `il eval` success-rate harnesses
  (success rate + grasp→lift→place funnel, dumped to JSON), success-rate and
  distance panels in `rl plot`, and success-based `model_best.pt` selection.
- **Tooling** — `uv`-based installer that resolves the isaaclab/lerobot
  dependency conflict, a GitHub-Release fetch script for the third-party
  SO-ARM101 USD mesh, and a CPU-only CI test suite.

### Known limitations / planned work
- The reward designs build, step, and pass their unit/runtime checks, but **no
  task has yet been trained to convergence**; the pick-place reward
  magnitudes are tuned analytically and need empirical validation.
- Stack lacks a shaped pre-release term; action-latency domain randomization is
  implemented but not yet wired into training; RL policies use privileged sim
  observations and so are not directly deployable to hardware (IL is).
- See the README and `docs/guides/` for the current sim-to-real caveats.

[0.1.0]: https://github.com/jixinyan/OpenSO-101/releases/tag/v0.1.0
