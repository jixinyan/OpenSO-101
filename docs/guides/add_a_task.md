# How to Add a New Task

This guide walks you through creating a new manipulation task for the SO-101
in Isaac Lab using the OpenSO-101 framework. Every task is a single
`OpenSO101EnvCfg` subclass registered with a single gym ID; variants
(cameras, teleop, play) are selected at `gym.make(...)` time via three
`configure_*` hooks on the cfg class.

> Legacy note: the Safe Sim2Real codebase used a multi-class-per-task pattern
> with separate `*EnvCfg`, `*EnvCfgWithCameras`, `*TeleopEnvCfg`, `*_PLAY`
> classes and a `joint_pos_env_cfg.py` glue module that registered four gym IDs
> per task. That pattern is documented for historical context in
> [`safe_sim2real/docs/how_to_design_a_task.md`](https://github.com/jixinyan/safe_sim2real/blob/main/docs/how_to_design_a_task.md).
> OpenSO-101 collapses those four classes into one and selects variants via
> kwargs to `gym.make`.

## Philosophy: single class, three variant hooks

In OpenSO-101, every task is exactly one `OpenSO101EnvCfg` subclass and is
registered under exactly one gym ID. Variants come from kwargs:

```python
import gymnasium as gym
import openso101.tasks  # noqa: F401   trigger gym registration

gym.make("OpenSO101-PickPlace-v0")                                    # state-only RL
gym.make("OpenSO101-PickPlace-v0", play=True)                         # eval / playback
gym.make("OpenSO101-PickPlace-v0", cameras=True)                      # RL with cameras
gym.make("OpenSO101-PickPlace-v0", action_mode="teleop", cameras=True)# teleop-vision
```

The framework routes each kwarg to a `configure_*` hook on the cfg before
constructing the env:

| Kwarg | Hook | Default behavior |
|---|---|---|
| `action_mode="rl"` (default) | `configure_action_mode("rl")` | No-op; uses the cfg's own `actions` manager. |
| `action_mode="teleop"` | `configure_action_mode("teleop")` | Swaps in absolute 6-joint position targets (`TeleopActionsCfg`). |
| `cameras=True` | `configure_cameras(True)` | Attaches `overhead_camera` + `wrist_camera` to `self.scene`. |
| `play=True` | `configure_play(True)` | Caps `num_envs` at 16 and disables observation corruption. |

If a task does not support a variant (for example, a vision-only task with no
meaningful teleop semantics), its `configure_*` hook should raise
`UnsupportedVariantError`. Callers see a clear, typed failure instead of a
silent misconfiguration.

## Project layout

A new task `MyLab-PourTea-v0` lives in your own package, not under
`openso101.tasks`. The layout follows the same convention as built-in tasks:

```text
mypkg/
  tasks/
    pour_tea/
      __init__.py                  @register_task wiring
      pour_tea_env_cfg.py          OpenSO101EnvCfg subclass + __post_init__
      mdp/
        __init__.py
        rewards.py
        terminations.py
        # observations.py only if shared helpers are insufficient
      agents/
        __init__.py
        rsl_rl_ppo_cfg.py
```

You must `import mypkg.tasks` (or any module that transitively imports it)
once before `gym.make` to trigger the registration side-effect.

## Step-by-step

### 1. Subclass `OpenSO101EnvCfg`

Define scene, actions, observations, rewards, terminations, and events in
`__post_init__`. Call `super().__post_init__()` first.

```python
# mypkg/tasks/pour_tea/pour_tea_env_cfg.py

from __future__ import annotations

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.mdp import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from openso101.envs import OpenSO101EnvCfg, UnsupportedVariantError
from openso101.robots import (
    SO101_DEFAULT_JOINT_POS,
    SO101_SIM_JOINT_NAMES,
    SO_ARM101_CFG,
)
from openso101.tasks.shared.objects import so101_cube_object_cfg


@configclass
class PourTeaSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ...        # filled in __post_init__
    teapot: RigidObjectCfg = ...        # filled in __post_init__
    cup: RigidObjectCfg = ...           # filled in __post_init__

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path="..."),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.5},
        close_command_expr={"gripper": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        teapot_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("teapot")},
        )
        cup_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("cup")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # plug in rewards from mypkg.tasks.pour_tea.mdp.rewards
    ...


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    teapot_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("teapot")},
    )


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class PourTeaEnvCfg(OpenSO101EnvCfg):
    """Pour the contents of a teapot into a cup."""

    scene: PourTeaSceneCfg = PourTeaSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM101_CFG.init_state.replace(joint_pos=SO101_DEFAULT_JOINT_POS),
        )
        self.scene.teapot = so101_cube_object_cfg(
            prim_path="{ENV_REGEX_NS}/Teapot",
            init_pos=(0.30, -0.05, 0.05),
            diffuse_color=(0.85, 0.20, 0.20),
        )
        self.scene.cup = so101_cube_object_cfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_pos=(0.30, 0.10, 0.03),
            diffuse_color=(0.20, 0.40, 0.85),
        )
```

### 2. Override `configure_*` only if needed

The base `OpenSO101EnvCfg` provides sensible defaults for all three hooks
(see `openso101/envs/base.py`). Override only when your task needs different
behavior — most commonly:

- Add task-specific camera prims in addition to the standard wrist+overhead pair.
- Reject `action_mode="teleop"` for tasks where teleop is meaningless.
- Customize what "play" mode disables.

```python
class PourTeaEnvCfg(OpenSO101EnvCfg):
    ...

    def configure_action_mode(self, mode: str) -> None:
        if mode == "teleop":
            raise UnsupportedVariantError(
                "PourTea has no teleop variant — the pour pose is procedural."
            )
        super().configure_action_mode(mode)

    def configure_cameras(self, enabled: bool) -> None:
        super().configure_cameras(enabled)
        if enabled:
            # Add a third side-view camera unique to this task.
            from isaaclab.sensors import TiledCameraCfg
            self.scene.side_camera = TiledCameraCfg(...)
```

If none of the three variants apply (rare), just leave the inherited methods
alone — the framework will pass `action_mode="rl"`, `cameras=False`,
`play=False` by default.

### 3. Wire the gym registration

The `register_task` decorator handles `gym.register` plus the variant-routing
env factory. Put this in your task's `__init__.py`:

```python
# mypkg/tasks/pour_tea/__init__.py

from openso101.envs import register_task

from .pour_tea_env_cfg import PourTeaEnvCfg

register_task(
    "MyLab-PourTea-v0",
    agent_cfgs={
        "rsl_rl_cfg_entry_point": "mypkg.tasks.pour_tea.agents.rsl_rl_ppo_cfg:PourTeaPPORunnerCfg",
    },
)(PourTeaEnvCfg)
```

You can pass any additional agent-cfg entry points through `agent_cfgs`; they
are merged into the gym spec's `kwargs` and become accessible via
`gym.spec("MyLab-PourTea-v0").kwargs`.

Defaults for the three variant kwargs are also bakeable into the spec:

```python
register_task(
    "MyLab-PourTea-v0",
    agent_cfgs={...},
    default_action_mode="rl",
    default_cameras=False,
    default_play=False,
)(PourTeaEnvCfg)
```

### 4. Write MDP functions

Standard Isaac Lab pattern: each function takes `env: ManagerBasedRLEnv` and
returns a `torch.Tensor` of shape `(num_envs,)`. Place them in
`mypkg/tasks/pour_tea/mdp/{rewards.py, terminations.py}` and import them
inside the `RewardsCfg` / `TerminationsCfg` blocks above.

For reward design guidance and anti-cheat patterns, study the existing
task implementations under `src/openso101/tasks/{lift,pick_place,stack}/`
— they cover controlled-lift gating, push-to-goal exploit prevention,
and the dense-chain reward (reach → grip-near → lift → goal-track) tuned
for SO-101's joint limits.

### 5. Configure the PPO runner

```python
# mypkg/tasks/pour_tea/agents/rsl_rl_ppo_cfg.py

from rsl_rl.modules import ActorCriticCfg as RslRlPpoActorCriticCfg
from rsl_rl.algorithms import PPOCfg as RslRlPpoAlgorithmCfg
from rsl_rl.runners import OnPolicyRunnerCfg as RslRlOnPolicyRunnerCfg
from isaaclab.utils import configclass


@configclass
class PourTeaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = "pour_tea"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        noise_std_type="log",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### 6. Run it

After installing your package (`pip install -e .`), make sure
`mypkg.tasks.pour_tea` is imported once — usually from `mypkg/tasks/__init__.py`:

```python
# mypkg/tasks/__init__.py
from . import pour_tea  # noqa: F401
```

Then verify and train:

```bash
# Confirm the task is registered. Add `import mypkg.tasks` to your entry point
# (or set OPENSO101_USER_TASKS=mypkg.tasks if your CLI supports it) so the
# side-effect runs before `openso101 envs list`.
openso101 envs list

# Smoke-test scene + action space without training.
openso101 envs zero --task MyLab-PourTea-v0 --play

# Short training smoke.
openso101 rl train \
  --task MyLab-PourTea-v0 \
  --algo ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 10

# Full training.
openso101 rl train --task MyLab-PourTea-v0 --algo ppo --headless

# Playback the latest checkpoint with cameras.
openso101 rl play \
  --task MyLab-PourTea-v0 \
  --checkpoint logs/rsl_rl/pour_tea/<run>/model_3000.pt \
  --with-cameras
```

## Tips

- **Start with the simplest reward chain that any random policy can earn from.**
  Distance-based tanh kernels (`1 - tanh(d / std)`) are the standard starter.
- **Make the success predicate stricter than the reward shaping.** PPO will
  otherwise find the easiest path to high reward (push, flick, bump) and skip
  the intended manipulation.
- **Use `play=True` for visual debugging.** It caps `num_envs` at 16 and
  disables observation noise, which is much friendlier for watching what the
  policy is actually doing.
- **Camera variants are free.** The shared `configure_cameras` hook means you
  never need to maintain a `*SceneCfgWithCameras` subclass.
- **Raise `UnsupportedVariantError` proudly.** A task that explicitly refuses a
  variant is much safer than one that silently misbehaves.

## See also

- `openso101.envs.base` — `OpenSO101EnvCfg`, `UnsupportedVariantError`, and
  the default `configure_*` implementations.
- `openso101.envs.registry` — the `register_task` decorator and env factory.
- `openso101.tasks.pick_place.pick_place_env_cfg` — a concrete reference for
  a multi-stage manipulation task in this pattern.
- Upstream [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/) —
  the underlying `ManagerBasedRLEnvCfg`, MDP cfg block, and event-manager
  concepts.

---

_Merged from `safe_sim2real/docs/how_to_design_a_task.md` (atomic-skills companion `how_to_add_an_atomic_skill.md` was dropped — atomic skills are out of scope for OpenSO-101)._
