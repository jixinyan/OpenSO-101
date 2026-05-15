# Isaac Sim and Isaac Lab Learning Guide for Safe Sim2Real

> _Ported from `safe_sim2real/docs/isaac_sim_learning_guide.md` during the OpenSO-101 v0.1.0 refactor. Code references updated; historical narrative preserved._

This guide is a project-specific path for learning Isaac Sim, Isaac Lab, and
RSL-RL through this repository. It is not meant to replace NVIDIA's official
documentation. It explains the pieces you must understand to create, train,
debug, and extend the Safe Sim2Real SO-ARM101 manipulation environments.

The repo currently targets:

- Isaac Lab `2.3.0`
- Isaac Sim through `isaaclab[all,isaacsim]`
- Python `3.11`
- PyTorch `2.7.0`
- RSL-RL for PPO training
- Gymnasium task registration
- Manager-based Isaac Lab RL environments

## 1. The Mental Model

There are three layers that are easy to confuse:

| Layer | What it does | Where this repo touches it |
| --- | --- | --- |
| Isaac Sim | The Omniverse simulator application: USD scene graph, rendering, PhysX, cameras, materials, GUI/headless app. | Launched by `isaaclab.app.AppLauncher` in scripts. Assets become USD prims under `/World`. |
| Isaac Lab | The robotics/RL framework on top of Isaac Sim: scenes, assets, actions, observations, rewards, events, terminations, commands, curriculum. | All task configs under `src/openso101/tasks/`. |
| RSL-RL | PPO/distillation training loop that consumes vectorized Isaac Lab envs. | `openso101 rl train` / `openso101 rl play` (driven by each task's `agents/rsl_rl_ppo_cfg.py`). |

In practice, you usually write Isaac Lab config code, not raw Isaac Sim code.
Isaac Sim is still underneath everything, so you need to understand prim paths,
USD assets, rendering, PhysX properties, and why the simulator must launch
before some imports.

## 2. Project Map

Start here:

```text
src/openso101/
  robots/
    so101/
      so_arm101.py              SO-ARM101 robot ArticulationCfg
      cameras.py                overhead + wrist TiledCameraCfg factories
      constants.py              joint names + gripper open/close constants
  cli/                          `openso101` top-level CLI (envs, rl, il, ...)
  envs/
    base.py                     OpenSO101EnvCfg + configure_* hooks
    registry.py                 register_task decorator
  rl/                           RSL-RL training/playback plumbing
  tasks/
    __init__.py                 imports built-in tasks (triggers gym registration)
    shared/                     shared objects, observations, rewards, terminations
    lift/
    pick_place/                 (was: composite/pick_and_place)
    stack/
  sim2real/
    domain_randomization/       shared physics DR helpers
docs/
  isaac_sim_learning_guide.md
  development_diary.md
  guides/
    teleop.md
    add_a_task.md
```

The highest-signal files to read first are:

1. `README.md`
2. `docs/guides/add_a_task.md`
3. `src/openso101/tasks/__init__.py`
4. `src/openso101/envs/base.py`
5. `src/openso101/envs/registry.py`
6. `src/openso101/tasks/pick_place/pick_place_env_cfg.py`
7. `src/openso101/tasks/lift/lift_env_cfg.py`
8. `src/openso101/cli/rl.py`
9. `src/openso101/cli/envs.py`

## 3. Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate openso101
python -m pip install -r requirements-cuda.txt
python -m pip install -e .
```

Why the install is split:

- `environment.yml` creates the Python environment.
- `requirements-cuda.txt` installs the GPU-correct PyTorch wheel first.
- `pip install -e .` registers this repo as an editable Python package.

Confirm basic package discovery:

```bash
openso101 envs list
```

If that works, the package can launch Isaac Sim, import `openso101.tasks`,
auto-register the tasks, and print the Gym IDs.

## 4. What Happens When You Run a Script

Every Isaac Lab script follows the same rough sequence:

1. Parse CLI args.
2. Set camera/rendering flags before launching the app.
3. Start Isaac Sim with `AppLauncher`.
4. Import Isaac Lab, Gymnasium, and this repo's task package.
5. Load a task config from the Gym registry through Hydra helpers.
6. Instantiate the environment with `gym.make(...)`.
7. Wrap it for RSL-RL if training or playback.
8. Step the simulation until training/playback ends.
9. Close the env and the simulator app.

In `train.py`, the important line is:

```python
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
```

By that point, the Gym ID has already resolved to an environment config class
and a PPO runner config class.

## 5. Gym Registration and Task Discovery

Task discovery starts at:

```python
# src/openso101/tasks/__init__.py
from . import lift        # noqa: F401
from . import pick_place  # noqa: F401
from . import stack       # noqa: F401
```

Each task subpackage's `__init__.py` calls `register_task(...)` to wire the
gym ID. Example pattern:

```python
from openso101.envs import register_task

from .pick_place_env_cfg import PickPlaceEnvCfg

register_task(
    "OpenSO101-PickPlace-v0",
    agent_cfgs={
        "rsl_rl_ppo_cfg_entry_point": "openso101.tasks.pick_place.agents.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)(PickPlaceEnvCfg)
```

That registration means:

- The simulator environment class is `ManagerBasedRLEnv` (wrapped by an
  `OpenSO101EnvCfg`-aware factory).
- The task behavior comes from the decorated `OpenSO101EnvCfg` subclass.
- The PPO settings come from `agent_cfgs["rsl_rl_ppo_cfg_entry_point"]`.
- Variants (cameras, teleop, play) are selected at `gym.make(...)` time via
  the `action_mode`, `cameras`, `play` kwargs.

Use this command whenever you add or rename a task:

```bash
openso101 envs list
```

## 6. Manager-Based Environments

This project uses Isaac Lab's manager-based environment style. You declare
configuration objects, and Isaac Lab builds managers from them.

A typical task config contains:

| Config class | Purpose |
| --- | --- |
| `SceneCfg` | What exists in the world: robot, table, objects, lights, cameras. |
| `CommandsCfg` | Goals or target signals sampled per episode or updated during the episode. |
| `ActionsCfg` | How policy actions map to robot controls. |
| `ObservationsCfg` | What tensors the policy receives. |
| `EventCfg` | Reset-time or startup randomization. |
| `RewardsCfg` | Reward terms and weights. |
| `TerminationsCfg` | Timeout, failure, and success conditions. |
| `CurriculumCfg` | Scheduled changes during training. |
| `EnvCfg` | Top-level scene, MDP managers, timestep, decimation, episode length. |

The pattern is visible in `PickPlaceEnvCfg`:

```python
@configclass
class PickPlaceEnvCfg(OpenSO101EnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 4.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
```

With `sim.dt = 0.01` and `decimation = 2`, physics runs at 100 Hz and the
policy acts every 2 physics steps, so the control rate is 50 Hz.

## 7. Scenes, Assets, and Prim Paths

Isaac Sim stores the world as USD prims. Isaac Lab config objects create those
prims for you.

Common asset config types:

- `ArticulationCfg`: robot with joints, e.g. SO-ARM101.
- `RigidObjectCfg`: movable rigid bodies, e.g. cubes.
- `AssetBaseCfg`: static assets, lights, ground plane, tables.
- `TiledCameraCfg`: GPU-friendly batched camera sensor.
- `FrameTransformerCfg`: computed frames such as the end-effector frame.

Safe Sim2Real uses per-environment prim paths:

```python
prim_path="{ENV_REGEX_NS}/Robot"
prim_path="{ENV_REGEX_NS}/Object"
prim_path="{ENV_REGEX_NS}/Table"
```

`{ENV_REGEX_NS}` is Isaac Lab's way of creating the same scene layout for many
parallel environments.

The robot is inserted in the task's `__post_init__`:

```python
from openso101.robots import SO_ARM101_CFG

self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

Objects are usually declared with `RigidObjectCfg` and spawned from primitive
geometry:

```python
self.scene.object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
    spawn=sim_utils.CuboidCfg(
        size=(0.03, 0.03, 0.03),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
)
```

## 8. Actions

Policies output normalized actions. Isaac Lab action terms turn those values
into robot targets.

The standard SO-ARM101 setup is:

```python
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
    scale=0.5,
    use_default_offset=True,
)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["gripper"],
    open_command_expr={"gripper": 0.5},
    close_command_expr={"gripper": 0.0},
)
```

This means:

- The arm action controls shoulder, elbow, and wrist joints by position.
- The gripper action is binary open/close.
- The policy action vector has one dimension per action-controlled joint/group.

If a policy jitters, inspect action scaling and the `action_rate` penalty before
changing the robot model.

## 9. Observations

Observations are declared as terms. Each term returns a tensor. With
`concatenate_terms = True`, Isaac Lab concatenates them into the policy input.

The atomic skills intentionally share a 24-dimensional observation layout:

```text
joint_pos                6
joint_vel                6
object_position          3
target_object_position   3
actions                  6
```

That shared shape is important because future sequencing can switch policies
without reshaping observations.

Composite tasks can use the same basic pattern, but their command/goal meaning
may differ. PickPlace, for example, uses a curriculum goal command so the target
position moves through lift, carry, and place stages.

## 10. Commands

Commands are target signals managed by Isaac Lab's command manager. They are
useful when the policy needs a goal as part of observation or reward.

Common command styles in this repo:

- `UniformPoseCommandCfg`: random pose target sampled at reset or intervals.
- `CurriculumGoalCommandCfg`: custom PickPlace command whose goal changes by
  stage inside one episode.

In PickPlace:

```python
object_pose = mdp.CurriculumGoalCommandCfg(
    asset_name="robot",
    object_name="object",
    resampling_time_range=(1e9, 1e9),
    lift_height=0.10,
    carry_height=0.15,
    place_goal=(0.20, 0.18, 0.02),
    advance_threshold=0.03,
)
```

The command manager exposes the active command to observations/rewards:

```python
goal_b = env.command_manager.get_command("object_pose")
```

## 11. Rewards

Reward functions live in each task's `mdp/rewards.py` or in shared helpers.
They take `env` and return one scalar per parallel env:

```python
def my_reward(env: ManagerBasedRLEnv, ...) -> torch.Tensor:
    return reward_per_env
```

The reward config wires a function, parameters, and a weight:

```python
stage_0_lift_goal = RewTerm(
    func=mdp.cube_to_curriculum_stage_goal,
    params={"stage": 0, "std": 0.18, "minimal_height": 0.04, "command_name": "object_pose"},
    weight=12.0,
)

stage_0_complete_bonus = RewTerm(
    func=mdp.stage_completion_bonus,
    params={"completed_stage": 0, "command_name": "object_pose"},
    weight=10.0,
)
```

Good reward design for this project:

- Start with dense rewards that the untrained policy can discover.
- Use tanh distance kernels: `1 - tanh(distance / std)`.
- Add fine-grained distance terms only near the goal.
- Keep regularizers small at first.
- Use curriculum to ramp penalties after the policy can do something useful.
- Make success stricter than reward shaping.
- For manipulation, prove the intended sequence, not only final object pose.

Important anti-cheat rule:

If the intended behavior is pick, carry, and place, do not reward only
`object_at_goal`. Require controlled lift, controlled carry, release, and
settled object state. Otherwise PPO can learn to push, bump, or flick the cube.

## 12. Terminations

Terminations define timeout, failures, and early success:

```python
time_out = DoneTerm(func=mdp.time_out, time_out=True)
object_dropping = DoneTerm(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
)
```

For training diagnostics, termination rates are as important as reward curves.
If a task never terminates except by timeout, the success predicate may be too
strict or missing. If it terminates too early, success may be too easy or a
failure condition may be misfiring.

## 13. Events and Reset Logic

Events run on `startup`, `reset`, or other supported modes. This repo uses
events for:

- Resetting the scene to default.
- Randomizing object spawn position.
- Setting privileged initial states for atomic skills.
- Domain randomization.

Example object reset:

```python
reset_object_position = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.1, 0.2), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    },
)
```

Use reset events to define the task distribution. If a policy fails only on
certain starts, check the event ranges before changing PPO.

## 14. Domain Randomization

Shared physics domain randomization lives in:

```text
src/openso101/sim2real/domain_randomization/
```

Current Phase 1 randomizes:

- Robot link masses.
- Robot joint friction and armature.
- Robot actuator PD gains.
- Object mass.
- Object contact material.
- Small gravity offsets.

Attach it inside a robot-specific task config:

```python
from openso101.sim2real.domain_randomization import attach_all_physics_dr

attach_all_physics_dr(
    self.events,
    robot_asset_name="robot",
    object_asset_names=("object",),
    include_gravity=True,
)
```

Project policy:

- Keep physics DR centralized.
- Wire task coverage explicitly.
- Add state/action robustness before visual randomization.
- Apply visual DR only to `-Vision` variants.
- Compare against no-DR baselines before widening ranges.

## 15. Cameras and Vision Variants

State-only tasks should stay fast. Camera tasks are opt-in.

OpenSO-101 registers one gym ID per task and selects variants via kwargs.
For example, every variant of the Lift task uses the same ID:

```text
OpenSO101-Lift-v0
```

```python
gym.make("OpenSO101-Lift-v0")                                            # RL, state-only
gym.make("OpenSO101-Lift-v0", play=True)                                 # eval / playback
gym.make("OpenSO101-Lift-v0", cameras=True)                              # RL with cameras
gym.make("OpenSO101-Lift-v0", cameras=True, play=True)                   # vision-play
gym.make("OpenSO101-Lift-v0", action_mode="teleop", cameras=True)        # teleop-vision
```

Camera factories live in:

```text
src/openso101/robots/so101/cameras.py
```

The shared `OpenSO101EnvCfg.configure_cameras(True)` attaches both cameras to
the scene; override it in your task cfg only if you need different geometry.
For reference, the base implementation is:

```python
def configure_cameras(self, enabled: bool) -> None:
    if not enabled:
        return
    self.scene.overhead_camera = overhead_camera_cfg()
    self.scene.wrist_camera = wrist_camera_cfg()
```

Preview camera frames:

```bash
openso101 envs preview \
  --task OpenSO101-Lift-v0 \
  --cameras \
  --output ./camera_preview \
  --headless
```

When the view matches the physical rig, copy the resulting position/quaternion
back into `cameras.py`.

## 16. Training

Train a policy:

```bash
openso101 rl train \
  --task OpenSO101-Lift-v0 \
  --algo ppo \
  --headless
```

Disable training videos when you want maximum throughput (videos are off by
default — opt in explicitly with `--video`):

```bash
openso101 rl train \
  --task OpenSO101-Lift-v0 \
  --algo ppo \
  --headless
```

Run a short smoke training:

```bash
openso101 rl train \
  --task OpenSO101-Lift-v0 \
  --algo ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 10
```

Resume:

```bash
openso101 rl train \
  --task OpenSO101-Lift-v0 \
  --algo ppo \
  --headless \
  --resume \
  --load_run 2026-04-24_12-00-00 \
  --checkpoint model_500.pt
```

Logs are written to:

```text
logs/rsl_rl/<experiment_name>/<run_timestamp>/
```

Each task's `agents/rsl_rl_ppo_cfg.py` controls:

- `num_steps_per_env`
- `max_iterations`
- `save_interval`
- `experiment_name`
- actor/critic network size
- entropy coefficient
- learning rate
- `gamma`
- `lam`

Atomic skills usually train with shorter episodes and smaller/simple reward
structures. Composite tasks often need longer rollouts, stronger curriculum,
and stricter success checks.

## 17. Playback and Export

Replay a specific checkpoint:

```bash
openso101 rl play \
  --task OpenSO101-Lift-v0 \
  --checkpoint logs/rsl_rl/lift/<run>/model_500.pt
```

Record playback video:

```bash
openso101 rl play \
  --task OpenSO101-Lift-v0 \
  --checkpoint logs/rsl_rl/lift/<run>/model_500.pt \
  --video \
  --video_length 300
```

The playback script exports policies to:

```text
logs/rsl_rl/<experiment>/<run>/exported/policy.pt
logs/rsl_rl/<experiment>/<run>/exported/policy.onnx
```

## 18. Plotting Training Curves

Plot the latest run for an experiment:

```bash
openso101 rl plot --task lift
```

Plot and save:

```bash
openso101 rl plot \
  --task lift \
  --save
```

Useful curves:

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Reward/<reward_term>`
- `Episode_Termination/<termination_term>`
- `Policy/mean_noise_std`
- PPO losses and entropy

If `Policy/mean_noise_std` collapses too early, the policy may stop exploring.
If mean episode length is always maximum, success may not be firing. If a
failure termination dominates, inspect reset distributions and thresholds.

## 19. Creating a New Atomic Skill

Use atomic skills for one primitive only:

- Grasp
- Lift
- Transport
- Place
- Return home
- A similarly short single-step behavior

> Note: atomic skills are a legacy concept from the Safe Sim2Real codebase.
> OpenSO-101 no longer ships an `atomic/` task family; every task is a single
> `OpenSO101EnvCfg` subclass following the pattern in
> [`docs/guides/add_a_task.md`](guides/add_a_task.md). The notes below are kept
> for historical context only.

Fastest workflow (legacy reference):

```bash
cp -r src/safe_sim2real/tasks/atomic/place src/safe_sim2real/tasks/atomic/my_new_skill
```

Then edit:

```text
src/safe_sim2real/tasks/atomic/my_new_skill/
  __init__.py
  my_new_skill_env_cfg.py
  joint_pos_env_cfg.py
  agents/rsl_rl_ppo_cfg.py
  mdp/__init__.py
  mdp/rewards.py
  mdp/terminations.py
```

Keep these invariants:

- One primitive, not a long sequence.
- Unified 24-dim observation space.
- 3-5 second episodes.
- 2-3 task rewards plus regularizers.
- Privileged initial state is allowed if the skill assumes previous skills
  already succeeded.
- Include state-only and `-Vision` registrations.
- Confirm the skill appears in `openso101 envs list`.

Atomic skill checklist (legacy):

```bash
openso101 envs list
openso101 envs zero --task <TaskGymId>
openso101 rl train \
  --task <TaskGymId> \
  --algo ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 10
```

## 20. Creating a New Composite Task

Use composite tasks for multi-stage behavior learned end-to-end:

- Reach
- Lift from free object
- Pick-and-place
- Stack
- Longer tabletop tasks

Directory shape:

```text
src/openso101/tasks/my_task/
  __init__.py                  # @register_task wiring
  my_task_env_cfg.py           # OpenSO101EnvCfg subclass + __post_init__
  agents/
    __init__.py
    rsl_rl_ppo_cfg.py
  mdp/
    __init__.py
    observations.py
    rewards.py
    terminations.py
```

Start from the closest existing task:

- Use `tasks/lift` for one object plus air goal.
- Use `tasks/pick_place` for staged manipulation.
- Use `tasks/stack` for multi-object manipulation.

Composite design order:

1. Define the task in plain English.
2. Define the success predicate first.
3. Define failure conditions.
4. Define object reset distributions.
5. Define the command/goal signal.
6. Define observations.
7. Define dense bootstrap rewards.
8. Add stricter gated rewards.
9. Add curriculum and regularizers.
10. Register state and vision variants.
11. Run zero/random agent smoke tests.
12. Run a short PPO smoke run.
13. Inspect reward and termination curves.
14. Only then run full training.

For pick/carry/place tasks, design against shortcut behavior from the start:

- Do not let pushing count as carrying.
- Do not let final object position alone count as success.
- Require gripper control before carry/place rewards activate.
- Require the object to settle at the goal after release.
- Penalize or terminate large uncontrolled XY motion when appropriate.

## 21. Creating a Task From Scratch

This is the full from-scratch flow without copying blindly.

### Step 1: Pick task family

OpenSO-101 ships a single task family. Every task is a single
`OpenSO101EnvCfg` subclass; variants (cameras, teleop, play) come from the
`configure_*` hooks. See [`docs/guides/add_a_task.md`](guides/add_a_task.md).

### Step 2: Create directories

```bash
mkdir -p src/openso101/tasks/my_task/agents
mkdir -p src/openso101/tasks/my_task/mdp
```

Create these files:

```text
__init__.py
my_task_env_cfg.py
agents/__init__.py
agents/rsl_rl_ppo_cfg.py
mdp/__init__.py
mdp/rewards.py
mdp/terminations.py
```

Add `mdp/observations.py` only if shared observation helpers are insufficient.

### Step 3: Write the base scene

Include placeholders for robot, end-effector frame, and movable objects:

```python
@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING
```

Add table, ground plane, and light. Do **not** attach cameras here — the
shared `OpenSO101EnvCfg.configure_cameras(True)` hook adds them when the
caller passes `cameras=True` to `gym.make`.

### Step 4: Write the MDP config classes

Create:

- `CommandsCfg`
- `ActionsCfg`
- `ObservationsCfg`
- `EventCfg`
- `RewardsCfg`
- `TerminationsCfg`
- `CurriculumCfg`

Use existing task configs as templates for imports and term names.

### Step 5: Write the robot-specific config

Inside the task `EnvCfg.__post_init__`:

- Call `super().__post_init__()` first.
- Set `self.scene.robot = SO_ARM101_CFG.replace(...)`.
- Set `arm_action`.
- Set `gripper_action`.
- Spawn objects.
- Define `FrameTransformerCfg` for `gripper_link`.
- Attach domain randomization if intended.
- Override `configure_cameras` / `configure_play` / `configure_action_mode`
  only when the defaults from `OpenSO101EnvCfg` don't fit.

### Step 6: Write MDP functions

Reward and termination functions should:

- Accept `env` first.
- Use `env.scene[...]` for assets.
- Return shape `(num_envs,)`.
- Avoid Python loops over envs.
- Use Torch tensor operations.
- Keep all tensors on the simulator device.

### Step 7: Register Gym ID

Register one ID — variants are selected per `gym.make()` call:

```python
@register_task("OpenSO101-MyTask-v0", agent_cfgs={...})
class MyTaskEnvCfg(OpenSO101EnvCfg):
    ...
```

```text
OpenSO101-MyTask-v0
```

### Step 8: Add PPO config

Start conservative:

- Short-horizon: `num_steps_per_env` around `24-32`, `max_iterations` around `1500`.
- Multi-stage: longer rollouts if one episode spans multiple stages.
- `actor_hidden_dims=[256, 128, 64]` is a good default.
- Use `entropy_coef` high enough to preserve exploration.

### Step 9: Verify registration and smoke behavior

```bash
openso101 envs list
openso101 envs zero --task OpenSO101-MyTask-v0 --play
openso101 envs random --task OpenSO101-MyTask-v0 --play
```

The zero/random agents run until the simulator app is closed or interrupted.
Use them to confirm that the scene loads, the action/observation spaces are
valid, and resets do not immediately fail.

### Step 10: Short training smoke

```bash
openso101 rl train \
  --task OpenSO101-MyTask-v0 \
  --algo ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 10
```

Do not start a full training run until the task resets cleanly, observations
have the expected shape, reward terms are nonzero when expected, and failure
terminations are not immediately firing.

## 22. Debugging Workflow

Use this order when a task fails:

1. `openso101 envs list`
   - Confirms registration and import errors.
2. Play with a small env count.
   - Confirms scene instantiation.
3. Run zero/random agent.
   - Confirms action space, reset, and stepping.
4. Run a 10-iteration PPO smoke.
   - Confirms train loop, logging, reward terms, and terminations.
5. Inspect TensorBoard or `plot_training.py`.
   - Confirms learning dynamics.
6. Watch playback.
   - Confirms the policy solves the intended behavior, not a shortcut.

Common failure modes:

| Symptom | Likely cause |
| --- | --- |
| Task missing from `list_envs` | Package not imported, registration typo, bad `env_cfg_entry_point`. |
| Isaac import fails before app launch | Import order issue; scripts that use Isaac Sim must launch with `AppLauncher` first. |
| Immediate object drop | Reset pose bad, object starts penetrating, gripper not actually holding object. |
| Reward always zero | Distance frame mismatch, wrong asset name, gate too strict. |
| Success never fires | Threshold too tight, wrong command frame, object not settled. |
| PPO learns pushing/flicking | Success predicate too weak or reward allows final-pose shortcut. |
| Vision task fails without cameras | Use `-Vision` task ID or ensure launcher sets `enable_cameras`. |
| Training is very slow | Videos/cameras enabled, too many envs for GPU, expensive sensors. |

## 23. Code Quality Checks

For lightweight validation before heavy Isaac runs:

```bash
python3 -m compileall -q src/openso101
git diff --check
```

For task-level validation, prefer the smallest command that exercises the path:

```bash
openso101 envs list
openso101 rl train \
  --task OpenSO101-Lift-v0 \
  --algo ppo \
  --headless \
  --num_envs 64 \
  --max_iterations 10
```

Full training is not a substitute for checking whether the reward/success logic
matches the intended physical behavior.

## 24. Suggested Learning Plan

Follow this sequence:

1. Run `openso101 envs list` and map Gym IDs to files.
2. Read `lift_env_cfg.py` and identify every manager config.
3. Read `openso101.envs.base.OpenSO101EnvCfg` and the `configure_*` hooks.
4. Run a `Lift` playback or random-agent smoke with `--play`.
5. Train `Lift` for 10 iterations with 64 envs.
6. Plot or inspect logs.
7. Read `pick_place_env_cfg.py` and understand the curriculum command.
8. Modify one reward weight and run a short smoke test.
9. Add one harmless observation/reward diagnostic term.
10. Create a tiny new task by copying `lift` (see `docs/guides/add_a_task.md`).
11. Add camera variants and preview images.
12. Attach domain randomization and compare short curves to no-DR.
13. Run a full training job only after all short checks make sense.

## 25. Glossary

| Term | Meaning in this repo |
| --- | --- |
| Env | A Gymnasium/Isaac Lab environment instance, usually many parallel copies. |
| Scene | The set of assets, sensors, lights, and prims in each parallel env. |
| Prim | A USD scene graph object path, e.g. `{ENV_REGEX_NS}/Robot`. |
| Articulation | A robot or multi-joint asset controlled by PhysX. |
| Rigid object | A movable single-body object such as a cube. |
| Manager | Isaac Lab component that owns observations, rewards, actions, etc. |
| MDP function | Python function that computes observation, reward, termination, event, or command logic. |
| Command | A goal signal sampled or updated by Isaac Lab and exposed to the policy/rewards. |
| Event | Startup/reset/randomization logic. |
| Decimation | Number of physics steps per policy step. |
| Play variant | Small-env, low-noise task variant for viewing/evaluating. |
| Vision variant | Task variant that instantiates overhead and wrist cameras. |
| DR | Domain randomization for sim-to-real robustness. |

## 26. What To Read Next

After this guide, read these in order:

1. `docs/guides/add_a_task.md`
2. `src/openso101/tasks/shared/`
3. `src/openso101/sim2real/domain_randomization/`
4. `src/openso101/cli/rl.py`
5. `src/openso101/cli/il.py`
6. The Isaac Lab manager-based environment tutorials.
7. Isaac Sim USD/PhysX/camera documentation.

The practical goal is not to memorize Isaac Sim. The goal is to understand the
path from a Gym ID to a configured scene, from policy action to robot movement,
from simulation state to rewards/terminations, and from logs to task-design
decisions.
