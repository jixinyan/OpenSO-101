# Migration from `safe_sim2real`

If you've been working with the legacy [`safe_sim2real`][safe-sim2real]
codebase, this page covers everything you need to migrate to OpenSO-101.

## Why migrate?

The OpenSO-101 refactor consolidates `safe_sim2real`'s eight overlapping
environment classes per task into one, replaces ad-hoc scripts with a
single `openso101` CLI, and reorganizes the package around four
pillars: RL, IL, Synthetic Data Generation, Sim-to-Real Robustness.

**Trade-off**: safe-RL (PPO-Lagrangian, CPO, FOCOPS) and the
SB3-StableBaselines3 wrappers are **not** ported. They stay in
`safe_sim2real`. If your research depends on them, keep using
`safe_sim2real` for those workflows and use OpenSO-101 for everything
else side-by-side.

## Quick Reference: Naming Changes

### Packages

| `safe_sim2real` | `openso101` |
|---|---|
| `safe_sim2real` (top-level) | `openso101` |
| `safe_sim2real.robots.trs_so101` | `openso101.robots.so101` |
| `safe_sim2real.robots.so101_constants` | `openso101.robots.so101.constants` |
| `safe_sim2real.tasks.composite.pick_and_place` | `openso101.tasks.pick_place` |
| `safe_sim2real.tasks.composite.lift` | `openso101.tasks.lift` |
| `safe_sim2real.tasks.composite.stack` | `openso101.tasks.stack` |
| `safe_sim2real.tasks.so101_object_cfg` | `openso101.tasks.shared.objects` |
| `safe_sim2real.tasks.so101_rl_defaults` | `openso101.tasks.shared.rl_defaults` |
| `safe_sim2real.tasks.domain_randomization` | `openso101.sim2real.domain_randomization.physics` |
| `safe_sim2real.teleop` | `openso101.teleop` (same module names) |

### Gym IDs

| `safe_sim2real` | `openso101` |
|---|---|
| `SafeSim2Real-SO-ARM101-PickPlace-v0` | `OpenSO101-PickPlace-v0` |
| `SafeSim2Real-SO-ARM101-PickPlace-Play-v0` | `gym.make("OpenSO101-PickPlace-v0", play=True)` |
| `SafeSim2Real-SO-ARM101-PickPlace-Vision-v0` | `gym.make("OpenSO101-PickPlace-v0", cameras=True)` |
| `SafeSim2Real-SO-ARM101-PickPlace-Teleop-v0` | `gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")` |
| `SafeSim2Real-SO-ARM101-Lift-Cube-v0` | `OpenSO101-Lift-v0` |
| `SafeSim2Real-SO-ARM101-Stack-Cube-v0` | `OpenSO101-Stack-v0` |
| `SafeSim2Real-SO-ARM101-PickPlace-Safe-PPOLag-v0` and other `-Safe-*` IDs | Not ported — stay in `safe_sim2real`. |

### Cfg Classes

| `safe_sim2real` | `openso101` |
|---|---|
| `SoArm101PickPlaceEnvCfg` | `PickPlaceEnvCfg` (+ kwargs to gym.make) |
| `SoArm101PickPlaceEnvCfg_PLAY` | `PickPlaceEnvCfg` + `configure_play(True)` |
| `SoArm101PickPlaceEnvCfgWithCameras` | `PickPlaceEnvCfg` + `configure_cameras(True)` |
| `SoArm101PickPlaceTeleopEnvCfg` | `PickPlaceEnvCfg` + `configure_action_mode("teleop")` |
| `SoArm101SafePickPlaceEnvCfg` | Not ported — `safe_sim2real` only |

The four legacy classes per task collapse into one class with three
`configure_*` hooks. See [`tasks_and_envs.md`](../concepts/tasks_and_envs.md).

### CLI Commands

| `safe_sim2real` | `openso101` |
|---|---|
| `python scripts/list_envs.py` | `openso101 envs list` |
| `python scripts/rsl_rl/train.py --task <id>` | `openso101 rl train --task <id> --algo ppo` |
| `python scripts/rsl_rl/play.py --task <id> --checkpoint <p>` | `openso101 rl play --task <id> --checkpoint <p>` |
| `python scripts/plot_training.py --task <name>` | `openso101 rl plot --task <name>` |
| `python scripts/lerobot/teleop_agent.py --task <id>` | `openso101 il record --task <id>` |
| `python scripts/lerobot/push_dataset.py --repo-root <d>` | `openso101 il push --repo-root <d>` |
| `python scripts/lerobot/replay_teleop_checkpoint.py --episode-path <p>` | `openso101 il replay --episode <p>` |
| `python scripts/preview_cameras.py --task <id>` | `openso101 envs preview --task <id>` |
| `python scripts/random_agent.py --task <id>` | `openso101 envs random --task <id>` |
| `python scripts/zero_agent.py --task <id>` | `openso101 envs zero --task <id>` |
| `python scripts/safe_rl/train.py ...` | Not ported — `safe_sim2real` only |
| `python scripts/sb3/train.py ...` | Not ported — out of scope |

## Step-by-Step Migration

### 1. Side-by-side checkout

You can have both repos installed in the same conda env:

```bash
cd ~/code
git clone https://github.com/KevinYan-831/safe_sim2real.git    # if not already
git clone https://github.com/KevinYan-831/OpenSO-101.git

conda activate safe_sim2real    # same env works for both
cd OpenSO-101 && pip install -e .
```

### 2. Update import statements

In your downstream code:

```diff
- from safe_sim2real.robots import SO_ARM101_CFG, SO101_SIM_JOINT_NAMES
+ from openso101.robots import SO_ARM101_CFG, SO101_SIM_JOINT_NAMES

- from safe_sim2real.tasks.so101_object_cfg import so101_cube_object_cfg
+ from openso101.tasks.shared.objects import so101_cube_object_cfg

- import safe_sim2real.tasks.composite.pick_and_place  # noqa
+ import openso101.tasks  # noqa
```

### 3. Update gym.make calls

```diff
- env = gym.make("SafeSim2Real-SO-ARM101-PickPlace-Vision-v0")
+ env = gym.make("OpenSO101-PickPlace-v0", cameras=True)

- env = gym.make("SafeSim2Real-SO-ARM101-PickPlace-Teleop-v0")
+ env = gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")
```

### 4. Update shell scripts / Makefiles

```diff
- python scripts/rsl_rl/train.py --task SafeSim2Real-SO-ARM101-PickPlace-v0 --headless
+ openso101 rl train --task OpenSO101-PickPlace-v0 --algo ppo --headless
```

### 5. Move teleop datasets

LeRobot datasets recorded with `safe_sim2real` are wire-compatible with
OpenSO-101 — only the embedded `task_name` metadata may differ. No
re-recording needed.

### 6. Keep safe-RL workflows in `safe_sim2real`

The two repos are siblings, not replacements. If you train both PPO and
PPO-Lagrangian policies, the recommended setup is:

- PPO + IL + teleop in OpenSO-101.
- PPO-Lagrangian / CPO / FOCOPS in safe_sim2real (no changes there).

### 7. Update your CI

Find:
```
pytest safe_sim2real/tests/
```
Replace with:
```
pytest /path/to/OpenSO-101/tests/
# OR if you've kept safe-RL tests too:
pytest safe_sim2real/tests/ /path/to/OpenSO-101/tests/
```

## Common Gotchas

**"My task subclass `SoArm101FooEnvCfg` doesn't exist anymore."** You
likely had a private task in `safe_sim2real` that you want to port.
Follow [`add_a_task.md`](add_a_task.md) — the migration boils down to
collapsing your four legacy classes into one with `configure_*` hooks.

**"My agent cfg `SoArm101SafePickPlaceEnvCfg`-style runner cfg is gone."**
That class is safe-RL and stays in `safe_sim2real`. Use
`PickPlacePPORunnerCfg` for the PPO baseline.

**"My old gym ID `...-Vision-v0` raises NameNotFound."** Expected: use
the kwarg form `gym.make("OpenSO101-PickPlace-v0", cameras=True)`.

## See Also

- [Tasks and envs](../concepts/tasks_and_envs.md)
- [Add a task](add_a_task.md)
- [`safe_sim2real` repo][safe-sim2real]

[safe-sim2real]: https://github.com/KevinYan-831/safe_sim2real
