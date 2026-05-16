# RL Algorithms

OpenSO-101 wires `Gymnasium`-registered Isaac Lab tasks to RL algorithms
through thin runner wrappers. Today the supported algorithms are
**PPO via [`rsl_rl`][rsl-rl]** and **Distillation** (teacher → student
knowledge transfer, also via rsl_rl). These are the only two algorithms
`rsl_rl` ships; off-policy methods (SAC, DDPG, TD3) would require a
different library entirely and are not currently on the roadmap.

## CLI

```bash
openso101 rl train --task <gym-id> --algo <name> [...]
```

The `--algo` flag selects the training stack:

| `--algo` | Status | Stack |
|---|---|---|
| `ppo` | ✅ supported | rsl_rl's `OnPolicyRunner` + `BestCheckpointRunner` |
| `distillation` | ✅ supported | rsl_rl's `DistillationRunner` (requires `--teacher-checkpoint`) |

Unknown algorithm names exit non-zero via argparse `choices`.

## PPO

**Backbone:** `rsl_rl.runners.OnPolicyRunner`, subclassed by
`openso101.rl.runners.BestCheckpointRunner` to also save `model_best.pt`
every time the 100-episode mean reward improves.

**Default hyperparameters** are defined in
`openso101/tasks/shared/rl_defaults.py` and shared by all built-in tasks:

```python
SO101_PPO_NUM_STEPS_PER_ENV
SO101_PPO_GAMMA
SO101_PPO_ENTROPY_COEF
SO101_PPO_INIT_NOISE_STD
SO101_PPO_NOISE_STD_TYPE
SO101_REACH_REWARD_COARSE_STD
SO101_REACH_REWARD_STD
```

Per-task runner cfgs (e.g. `PickPlacePPORunnerCfg`) live in
`openso101/tasks/<task>/agents/rsl_rl_ppo_cfg.py` and pull these
constants. To override a hyperparameter for a single task, edit only that
file; to override globally, edit `rl_defaults.py`.

**Logging:** `--logger tensorboard` (default), `wandb` (requires
`pip install -e ".[wandb]"`), or `neptune`.

**Resume:** `--resume --load_run <run-name>` or `--checkpoint <path>`.

## Distillation (teacher → student)

`rsl_rl` ships a `DistillationRunner` that takes a trained teacher and
trains a student to mimic its action distribution. Both networks share
the same MLP architecture by default (see
`SO101_DISTILL_HIDDEN_DIMS` in `rl_defaults.py`) so a freshly-trained
PPO checkpoint loads as a teacher without modification.

```bash
openso101 rl train \
  --task OpenSO101-PickPlace-v0 \
  --algo distillation \
  --teacher-checkpoint logs/rsl_rl/pick_place/<teacher-run-dir> \
  --headless
```

`--teacher-checkpoint` accepts either a run directory (rsl_rl finds the
latest `.pt` inside `checkpoints/`) or a direct `.pt` path. Internally
the CLI splits this into `agent_cfg.load_run` + `agent_cfg.load_checkpoint`
so rsl_rl's existing teacher-loading path is reused.

**Per-task config:** `openso101/tasks/<task>/agents/rsl_rl_distillation_cfg.py`
sets the iteration budget and any architecture overrides. All three
built-in tasks (Lift, PickPlace, Stack) ship a distillation cfg.

**When to use it:**
- You trained a PPO teacher with privileged observations (e.g. true
  cube pose, ground-truth velocities) and want a deployable student
  that only sees realistic observations.
- You want a smaller / faster student than the teacher for real-time
  deployment.

## Play / Replay

```bash
openso101 rl play --task <gym-id> --checkpoint logs/.../model_best.pt
```

Loads the policy and runs it in a single-env playback session. `--with-cameras`
opens wrist and overhead viewports.

## Plotting

```bash
openso101 rl plot --task pick_place --smooth 30
# or
openso101 rl plot --log_dir logs/rsl_rl/pick_place/<run> --save
```

The plotter auto-picks the latest run for a given task name, or you can
point at an exact run directory. `--save` writes a PNG instead of opening
a window.

## Algorithm Roadmap

The PPO + Distillation pair covers the canonical Isaac-Sim
on-policy-then-distill workflow. Off-policy methods (SAC, DDPG, TD3) are
not in `rsl_rl` and would require integrating a separate library
(`torchrl`, `stable-baselines3`, or in-house) behind a new
`openso101.rl.algos.*` module — not currently planned.

[rsl-rl]: https://github.com/leggedrobotics/rsl_rl
