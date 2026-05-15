# RL Algorithms

OpenSO-101 wires `Gymnasium`-registered Isaac Lab tasks to RL algorithms
through thin runner wrappers. Today the supported algorithm is **PPO via
[`rsl_rl`][rsl-rl]**; **SAC** (off-policy) lands with sub-project E.
**Safe-RL is intentionally out of scope** — it stays in the legacy
[`safe_sim2real`][safe-sim2real] repo.

## CLI

```bash
openso101 rl train --task <gym-id> --algo <name> [...]
```

The `--algo` flag selects the training stack:

| `--algo` | Status | Stack |
|---|---|---|
| `ppo` | ✅ supported | rsl_rl's `OnPolicyRunner` + `BestCheckpointRunner` |
| `sac` | 🛠️ planned (sub-project E) | rsl_rl off-policy runner |
| `ppo_lag` / `cpo` / `focops` | ❌ not in OpenSO-101 | Lives in `safe_sim2real` |

Unsupported algorithms exit `2` with a clear message; they are not silently
mapped to a different algorithm.

## PPO

The default and only fully-functional algorithm today.

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

**Logging:** `--logger wandb` (default), `tensorboard`, or `neptune`.

**Resume:** `--resume --load_run <run-name>` or `--checkpoint <path>`.

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
openso101 rl plot --log_dir logs/rsl_rl/openso101_pickplace/<run> --save
```

The plotter auto-picks the latest run for a given task name, or you can
point at an exact run directory. `--save` writes a PNG instead of opening
a window.

## Algorithm Roadmap

- **SAC** (sub-project E): off-policy actor-critic. Will reuse
  `openso101.rl.runners.OffPolicyRunner` (currently a skeleton).
- **Distributional / model-based**: out of scope for v0.1; reach out if
  you have a research case.
- **Safe-RL**: intentionally **not** in OpenSO-101. PPO-Lagrangian, CPO,
  and FOCOPS implementations live in [`safe_sim2real`][safe-sim2real] for
  the original `manipulation-with-constraints` line of work.

[rsl-rl]: https://github.com/leggedrobotics/rsl_rl
[safe-sim2real]: https://github.com/KevinYan-831/safe_sim2real
