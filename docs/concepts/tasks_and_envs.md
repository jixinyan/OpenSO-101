# Tasks and Environments

OpenSO-101 builds on Isaac Lab's `ManagerBasedRLEnvCfg`. This page explains the
small set of conventions that turn a stock Isaac Lab cfg into an OpenSO-101
task: the **single-class-per-task pattern**, the three **variant hooks**, and
the **`@register_task`** decorator.

## Task vs. Env

In Isaac Lab terminology:

- A **task** is the high-level intent (lift a cube, place a cube at a goal,
  stack two cubes).
- An **env** is a concrete Isaac Lab `ManagerBasedRLEnv` instance — robot,
  scene, actions, observations, rewards, terminations, and (optionally)
  domain randomization.

Every OpenSO-101 task ships exactly one **env config class** (e.g.
`PickPlaceEnvCfg`) and exactly one **gym ID** (`OpenSO101-PickPlace-v0`).
Variants of that task — RL training vs. teleop recording, state-only vs.
cameras, full env count vs. play mode — are produced by calling
`gym.make(...)` with kwargs on the same gym ID.

## The Single-Class-Per-Task Pattern

Legacy `safe_sim2real` exposed four classes per task:

```
SoArm101PickPlaceEnvCfg               # base RL
SoArm101PickPlaceEnvCfg_PLAY          # eval / playback (fewer envs)
SoArm101PickPlaceEnvCfgWithCameras    # state + wrist + overhead camera
SoArm101PickPlaceTeleopEnvCfg         # teleop action mode
```

Plus four matching gym IDs (`...-v0`, `...-Play-v0`, `...-Vision-v0`,
`...-Teleop-v0`) — eight by the time you add `-Vision-Play-v0` etc.

OpenSO-101 collapses all of those into one class and one gym ID:

```python
import gymnasium as gym
import openso101.tasks  # noqa: F401   trigger gym registration

gym.make("OpenSO101-PickPlace-v0")                                       # base RL
gym.make("OpenSO101-PickPlace-v0", play=True)                            # eval
gym.make("OpenSO101-PickPlace-v0", cameras=True)                         # +cameras
gym.make("OpenSO101-PickPlace-v0", action_mode="teleop", cameras=True)   # teleop+cameras
```

The kwargs are interpreted by three **variant hooks** on `OpenSO101EnvCfg`:

```python
class OpenSO101EnvCfg(ManagerBasedRLEnvCfg):
    def configure_action_mode(self, mode: str) -> None: ...
    def configure_cameras(self, enabled: bool) -> None: ...
    def configure_play(self, enabled: bool) -> None: ...
```

The registry adapter calls them in that order on a fresh cfg instance before
constructing the env. A subclass overrides one or more hooks; an
unsupported variant raises `UnsupportedVariantError`.

## Writing a Custom Task

Minimal cfg:

```python
from openso101.envs import OpenSO101EnvCfg, register_task, UnsupportedVariantError
from openso101.robots import SO_ARM101_CFG, SO_ARM101_TELEOP_CFG
from openso101.tasks.shared.objects import so101_cube_object_cfg


@register_task(
    "MyLab-PourTea-v0",
    agent_cfgs={
        "rsl_rl_cfg_entry_point":
            "mypkg.tasks.pour_tea.agents.rsl_rl_ppo_cfg:PourTeaPPORunnerCfg",
    },
)
class PourTeaCfg(OpenSO101EnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SO_ARM101_CFG
        self.scene.object = so101_cube_object_cfg(name="teacup")
        # ... actions, observations, rewards, terminations ...

    def configure_action_mode(self, mode: str) -> None:
        if mode == "teleop":
            self.scene.robot = SO_ARM101_TELEOP_CFG
            self.actions = TeleopActionsCfg()
        elif mode != "rl":
            raise UnsupportedVariantError(f"PourTea does not support {mode!r}")

    def configure_cameras(self, enabled: bool) -> None:
        if enabled:
            self.scene.wrist_camera = wrist_camera_cfg()
            self.scene.overhead_camera = overhead_camera_cfg()

    def configure_play(self, enabled: bool) -> None:
        if enabled:
            self.scene.num_envs = 4
            self.events = None  # disable domain randomization
```

Three points:
1. **`@register_task`** registers the class under the given gym ID and wires
   the rsl_rl PPO cfg entry point.
2. **`__post_init__`** sets up the base RL configuration.
3. **Three `configure_*` hooks** specialize for variants. Each is a no-op by
   default; override only the ones your task supports.

For a fully-worked example, see [`docs/guides/add_a_task.md`](../guides/add_a_task.md).

## Built-in Tasks

| Gym ID | Description | Variants |
|---|---|---|
| `OpenSO101-Lift-v0` | Lift a cube to a target height | rl, teleop, cameras, play |
| `OpenSO101-PickPlace-v0` | Pick up a cube and place it at a goal pose | rl, teleop, cameras, play |
| `OpenSO101-Stack-v0` | Stack one cube on top of another | rl, teleop, cameras, play |

All three are registered automatically when you `import openso101.tasks`
(which happens on `import openso101` if Isaac Lab is available).

## Discovery

To see the gym IDs registered in your environment:

```bash
openso101 envs list
```

To open one with cameras and a random policy as a smoke test:

```bash
openso101 envs random --task OpenSO101-PickPlace-v0 --with-cameras --steps 200
```
