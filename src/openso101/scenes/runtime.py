# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch
from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

from openso101.envs.base import OpenSO101EnvCfg, TeleopActionsCfg
from openso101.robots.so101.so_arm101 import SO101_USD_TABLETOP_ROOT_Z, SO_ARM101_CFG

from .models import Goal, SceneSpec
from .usd import verify_compilation


def scene_states(env):
    values = []
    for entity in env.cfg.scene_spec.entities:
        if entity.dynamic:
            state = env.scene[entity.entity_id].data.root_state_w.clone()
            state[:, :3] -= env.scene.env_origins
        else:
            state = torch.tensor((*entity.pose.position, *entity.pose.quaternion_wxyz, 0, 0, 0, 0, 0, 0), device=env.device)
            state = state.expand(env.num_envs, -1)
        values.append(state)
    return torch.stack(values, dim=1)


def object_observations(env):
    return scene_states(env).flatten(1)


def goal_observations(env):
    return torch.tensor(env.cfg.scene_spec.task.goal_position_m, device=env.device).expand(env.num_envs, -1)


def reset_objects(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    mdp.reset_scene_to_default(env, env_ids)
    for entity in env.cfg.scene_spec.entities:
        if entity.dynamic:
            obj = env.scene[entity.entity_id]
            state = obj.data.default_root_state[env_ids].clone()
            lower, upper = torch.tensor(entity.reset_translation_m, device=env.device)
            state[:, :3] += lower + torch.rand((len(env_ids), 3), device=env.device) * (upper - lower)
            state[:, :3] += env.scene.env_origins[env_ids]
            obj.write_root_state_to_sim(state, env_ids)
    if hasattr(env, "_scene_hold_seconds"):
        env._scene_hold_seconds[env_ids] = 0
        env._scene_success[env_ids] = False


def task_success(env):
    if not hasattr(env, "_scene_hold_seconds"):
        env._scene_hold_seconds = torch.zeros(env.num_envs, device=env.device)
        env._scene_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._scene_success_step = -1
    if env._scene_success_step == env.common_step_counter:
        return env._scene_success
    spec = env.cfg.scene_spec
    values = scene_states(env)
    states = {entity.entity_id: values[:, index] for index, entity in enumerate(spec.entities)}
    entities = {entity.entity_id: entity for entity in spec.entities}
    jaw_id = env.scene["robot"].joint_names.index("Jaw")
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    goals = spec.task.goals or (Goal(object_id=spec.task.object_id, position_m=spec.task.goal_position_m),)
    for goal in goals:
        state = states[goal.object_id]
        if goal.predicate == "at":
            position = torch.tensor(goal.position_m, device=env.device)
            reached = torch.linalg.vector_norm(state[:, :3] - position, dim=-1) <= spec.task.position_tolerance_m
        else:
            target = states[goal.target_id]
            local = quat_apply_inverse(target[:, 3:7], state[:, :3] - target[:, :3])
            relative = matrix_from_quat(target[:, 3:7]).transpose(-1, -2) @ matrix_from_quat(state[:, 3:7])
            half = torch.tensor(entities[goal.object_id].dimensions_m, device=env.device) / 2
            extent = relative.abs() @ half
            if goal.predicate == "inside":
                lower, upper = torch.tensor(goal.region_bounds_m, device=env.device)
                reached = ((local - extent >= lower) & (local + extent <= upper)).all(dim=-1)
            else:
                target_half = torch.tensor(entities[goal.target_id].dimensions_m, device=env.device) / 2
                reached = (local[:, :2].abs() + extent[:, :2] <= target_half[:2]).all(dim=-1)
                reached &= (local[:, 2] - extent[:, 2] - target_half[2]).abs() <= spec.task.position_tolerance_m
        stable = torch.linalg.vector_norm(state[:, 7:10], dim=-1) <= spec.task.max_linear_speed_m_s
        stable &= torch.linalg.vector_norm(state[:, 10:13], dim=-1) <= spec.task.max_angular_speed_rad_s
        valid &= reached & stable
    if spec.task.require_released:
        valid &= env.scene["robot"].data.joint_pos[:, jaw_id] > 0.4
    env._scene_hold_seconds = torch.where(valid, env._scene_hold_seconds + env.step_dt, 0)
    env._scene_success = env._scene_hold_seconds >= env.cfg.scene_spec.task.settle_seconds
    env._scene_success_step = env.common_step_counter
    return env._scene_success


def task_failure(env):
    values = scene_states(env)
    return (~torch.isfinite(values).all(dim=(1, 2))) | (values[:, :, 2] < env.cfg.scene_spec.table.top_z_m - 0.1).any(dim=1)


def task_reward(env):
    robot = env.scene["robot"]
    gripper = robot.data.body_pos_w[:, robot.body_names.index("gripper")]
    spec = env.cfg.scene_spec
    values = scene_states(env)
    states = {entity.entity_id: values[:, index] for index, entity in enumerate(spec.entities)}
    entities = {entity.entity_id: entity for entity in spec.entities}
    goals = spec.task.goals or (Goal(object_id=spec.task.object_id, position_m=spec.task.goal_position_m),)
    score = torch.zeros(env.num_envs, device=env.device)
    for goal in goals:
        obj = states[goal.object_id][:, :3]
        if goal.predicate == "at":
            position = torch.tensor(goal.position_m, device=env.device)
        else:
            target = states[goal.target_id]
            local = torch.zeros((env.num_envs, 3), device=env.device)
            if goal.predicate == "inside":
                local[:] = torch.tensor(goal.region_bounds_m, device=env.device).mean(dim=0)
            else:
                local[:, 2] = (entities[goal.target_id].dimensions_m[2] + entities[goal.object_id].dimensions_m[2]) / 2
            position = target[:, :3] + (matrix_from_quat(target[:, 3:7]) @ local.unsqueeze(-1)).squeeze(-1)
        approach = 1 - torch.tanh(torch.linalg.vector_norm(gripper - env.scene.env_origins - obj, dim=-1) / 0.1)
        progress = 1 - torch.tanh(torch.linalg.vector_norm(position - obj, dim=-1) / 0.1)
        score += approach + 2 * progress
    return score / len(goals) + 10 * task_success(env).float()


@configclass
class SceneObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        objects = ObservationTermCfg(func=object_observations)
        goal = ObservationTermCfg(func=goal_observations)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class SceneEventsCfg:
    reset_scene = EventTermCfg(func=reset_objects, mode="reset")


@configclass
class SceneTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=task_success)
    failure = TerminationTermCfg(func=task_failure)


@configclass
class SceneRewardsCfg:
    task = RewardTermCfg(func=task_reward, weight=1.0)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class SceneActionsCfg:
    joints = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CustomSceneEnvCfg(OpenSO101EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    observations: SceneObservationsCfg = SceneObservationsCfg()
    actions: SceneActionsCfg = SceneActionsCfg()
    events: SceneEventsCfg = SceneEventsCfg()
    rewards: SceneRewardsCfg = SceneRewardsCfg()
    terminations: SceneTerminationsCfg = SceneTerminationsCfg()
    scene_spec: SceneSpec | None = None
    compiled_scene: str = ""

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000))

    def configure_scene(self, path):
        folder = Path(path).resolve()
        verify_compilation(folder)
        if not (folder / "environment.usda").is_file():
            raise ValueError("场景缺少 environment.usda，请重新编译 scene bundle")
        self.scene_spec = spec = SceneSpec.read(folder / "scene.json")
        self.compiled_scene = str(folder)
        self.scene.content = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Content", spawn=sim_utils.UsdFileCfg(usd_path=str(folder / "environment.usda")),
        )
        for entity in spec.entities:
            if entity.entity_id in ("robot", "light", "content", "overhead_camera", "wrist_camera"):
                raise ValueError(f"entity_id 与系统实体重复：{entity.entity_id}")
            if entity.dynamic:
                setattr(self.scene, entity.entity_id, RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Content/Objects/{entity.entity_id}",
                    spawn=None, init_state=RigidObjectCfg.InitialStateCfg(pos=entity.pose.position, rot=entity.pose.quaternion_wxyz),
                ))
        self.scene.robot.init_state.pos = (
            spec.robot_base.position[0], spec.robot_base.position[1],
            spec.robot_base.position[2] + SO101_USD_TABLETOP_ROOT_Z,
        )
        self.scene.robot.init_state.rot = spec.robot_base.quaternion_wxyz
        self.seed = spec.reset_seed

    def configure_action_mode(self, mode):
        super().configure_action_mode(mode)
        if mode == "teleop":
            self.actions = TeleopActionsCfg()
            self.episode_length_s = 3600
            self.terminations.success = None


def register_custom_scene():
    from openso101.envs import register_task

    register_task("OpenSO101-CustomScene-v0", agent_cfgs={
        "rsl_rl_cfg_entry_point": "openso101.tasks.pick_place.agents.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    })(CustomSceneEnvCfg)
