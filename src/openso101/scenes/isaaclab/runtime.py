# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

from openso101.envs.base import OpenSO101EnvCfg, TeleopActionsCfg
from openso101.robots.so101.so_arm101 import SO101_USD_TABLETOP_ROOT_Z, SO_ARM101_CFG

from ..models import Goal, SceneSpec
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
    if not hasattr(env, "_scene_hold_seconds"):
        env._scene_hold_seconds = torch.zeros(env.num_envs, device=env.device)
        env._scene_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._scene_success_step = -1
    env._scene_hold_seconds[env_ids] = 0
    env._scene_success[env_ids] = False
    env._scene_success_step = -1
    if hasattr(env, "_bddl_trackers"):
        for index in env_ids.tolist():
            env._bddl_trackers[index].reset()
    if env.cfg.task_program is not None:
        if not hasattr(env, "_scene_program_phase"):
            env._scene_program_phase = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
            env._scene_program_hold = torch.zeros(env.num_envs, device=env.device)
        env._scene_program_phase[env_ids] = 0
        env._scene_program_hold[env_ids] = 0


def scene_jaw_forces(env):
    dynamic = [entity.entity_id for entity in env.cfg.scene_spec.entities if entity.dynamic]
    magnitudes = []
    for name in ("gripper_jaw_contact", "moving_jaw_contact"):
        matrix = env.scene[name].data.force_matrix_w
        if matrix.shape != (env.num_envs, 1, len(dynamic), 3) or not torch.isfinite(matrix).all():
            raise RuntimeError("场景双侧接触力的数量或数值无效")
        magnitudes.append(torch.linalg.vector_norm(matrix[:, 0], dim=-1))
    values = torch.stack(magnitudes, dim=-1)
    return {name: values[:, index] for index, name in enumerate(dynamic)}


def goal_reached(spec, goal, states, device):
    state = states[goal.object_id]
    entities = {entity.entity_id: entity for entity in spec.entities}
    if goal.predicate == "at":
        position = torch.tensor(goal.position_m, device=device)
        return torch.linalg.vector_norm(state[:, :3] - position, dim=-1) <= spec.task.position_tolerance_m
    target = states[goal.target_id]
    local = quat_apply_inverse(target[:, 3:7], state[:, :3] - target[:, :3])
    relative = matrix_from_quat(target[:, 3:7]).transpose(-1, -2) @ matrix_from_quat(state[:, 3:7])
    extent = relative.abs() @ (torch.tensor(entities[goal.object_id].dimensions_m, device=device) / 2)
    if goal.predicate == "inside":
        lower, upper = torch.tensor(goal.region_bounds_m, device=device)
        return ((local - extent >= lower) & (local + extent <= upper)).all(dim=-1)
    target_half = torch.tensor(entities[goal.target_id].dimensions_m, device=device) / 2
    reached = (local[:, :2].abs() + extent[:, :2] <= target_half[:2]).all(dim=-1)
    return reached & ((local[:, 2] - extent[:, 2] - target_half[2]).abs() <= spec.task.position_tolerance_m)


def program_condition(env, condition, states, forces, opened):
    program = env.cfg.task_program
    state = states[condition.object_id]
    if condition.kind == "grasped":
        return (forces[condition.object_id] > program.grasp_force_threshold_newtons).all(dim=-1)
    if condition.kind == "released":
        return opened & (forces[condition.object_id] <= program.release_force_threshold_newtons).all(dim=-1)
    if condition.kind == "lifted":
        return state[:, 2] - env.cfg.scene_spec.table.top_z_m >= condition.height_above_table_m
    if condition.kind == "stable":
        return ((torch.linalg.vector_norm(state[:, 7:10], dim=-1) <= env.cfg.scene_spec.task.max_linear_speed_m_s)
                & (torch.linalg.vector_norm(state[:, 10:13], dim=-1) <= env.cfg.scene_spec.task.max_angular_speed_rad_s))
    return goal_reached(env.cfg.scene_spec, condition.goal, states, env.device)


def program_progress(env):
    if not hasattr(env, "_scene_program_phase"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    return torch.stack((env._scene_program_phase.float() / len(env.cfg.task_program.phases),
                        env._scene_program_hold, env._scene_hold_seconds), dim=-1)


def bddl_trackers(env):
    if env.cfg.bddl_report is None:
        return ()
    if not hasattr(env, "_bddl_trackers"):
        from ..bddl import BDDLBinding, BDDLTaskTracker

        report = env.cfg.bddl_report
        binding = BDDLBinding.model_validate(report["binding"])
        env._bddl_trackers = [BDDLTaskTracker(report["problem"], binding, env.cfg.scene_spec)
                              for _ in range(env.num_envs)]
    return env._bddl_trackers


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
    jaw_id = env.scene["robot"].joint_names.index("Jaw")
    opened = env.scene["robot"].data.joint_pos[:, jaw_id] > .4
    forces = scene_jaw_forces(env)
    bddl_valid = None
    if env.cfg.bddl_report is not None:
        trackers = bddl_trackers(env)
        measured = {name: state.detach().cpu().numpy() for name, state in states.items()}
        opened_values = opened.detach().cpu().tolist()
        results = [tracker.update({name: state[index] for name, state in measured.items()},
                                  opened_values[index], env.step_dt)
                   for index, tracker in enumerate(trackers)]
        bddl_valid = torch.tensor([item["source_goal_satisfied"] and item["stable"] for item in results],
                                 device=env.device)
        if spec.task.require_released:
            bddl_valid &= opened
            for subject in env._bddl_trackers[0].subject_ids:
                bddl_valid &= (forces[subject] <= .1).all(dim=-1)
    if env.cfg.task_program is not None:
        if not hasattr(env, "_scene_program_phase"):
            env._scene_program_phase = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
            env._scene_program_hold = torch.zeros(env.num_envs, device=env.device)
        program = env.cfg.task_program
        active = env._scene_program_phase.clone()
        for index, phase in enumerate(program.phases):
            selected = active == index
            eligible = torch.stack([program_condition(env, item, states, forces, opened)
                                    for item in phase.conditions]).all(dim=0)
            duration = torch.where(eligible, env._scene_program_hold + env.step_dt, 0.)
            passed = selected & eligible & (duration + 1e-7 >= phase.hold_seconds)
            env._scene_program_hold[selected] = duration[selected]
            env._scene_program_hold[passed] = 0
            env._scene_program_phase[passed] += 1
        final = torch.stack([program_condition(env, item, states, forces, opened)
                             for item in program.final_conditions]).all(dim=0)
        eligible = (env._scene_program_phase == len(program.phases)) & final
        if bddl_valid is not None:
            eligible &= bddl_valid
        env._scene_hold_seconds = torch.where(eligible, env._scene_hold_seconds + env.step_dt, 0.)
        env._scene_success = env._scene_hold_seconds + 1e-7 >= program.settle_seconds
        env._scene_success_step = env.common_step_counter
        return env._scene_success
    if bddl_valid is not None:
        valid = bddl_valid
        env._scene_hold_seconds = torch.where(valid, env._scene_hold_seconds + env.step_dt, 0.)
        env._scene_success = env._scene_hold_seconds + 1e-7 >= spec.task.settle_seconds
        env._scene_success_step = env.common_step_counter
        return env._scene_success
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    goals = spec.task.goals or (Goal(object_id=spec.task.object_id, position_m=spec.task.goal_position_m),)
    for goal in goals:
        state = states[goal.object_id]
        reached = goal_reached(spec, goal, states, env.device)
        stable = torch.linalg.vector_norm(state[:, 7:10], dim=-1) <= spec.task.max_linear_speed_m_s
        stable &= torch.linalg.vector_norm(state[:, 10:13], dim=-1) <= spec.task.max_angular_speed_rad_s
        valid &= reached & stable
        if spec.task.require_released:
            valid &= (forces[goal.object_id] <= .1).all(dim=-1)
    if spec.task.require_released:
        valid &= opened
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
        task_program = None

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
    task_program: object | None = None
    bddl_report: dict | None = None

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
        self.bddl_report = None
        self.task_program = None
        self.observations.policy.task_program = None
        if (folder / "bddl.json").is_file():
            self.bddl_report = json.loads((folder / "bddl.json").read_text())
        self.scene.content = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Content", spawn=sim_utils.UsdFileCfg(usd_path=str(folder / "environment.usda")),
        )
        for entity in spec.entities:
            if entity.entity_id in ("robot", "light", "content", "overhead_camera", "wrist_camera",
                                    "gripper_jaw_contact", "moving_jaw_contact"):
                raise ValueError(f"entity_id 与系统实体重复：{entity.entity_id}")
            if entity.dynamic:
                setattr(self.scene, entity.entity_id, RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Content/Objects/{entity.entity_id}",
                    spawn=None, init_state=RigidObjectCfg.InitialStateCfg(pos=entity.pose.position, rot=entity.pose.quaternion_wxyz),
                ))
        filters = [f"{{ENV_REGEX_NS}}/Content/Objects/{entity.entity_id}"
                   for entity in spec.entities if entity.dynamic]
        for name, body in (("gripper_jaw_contact", "gripper"), ("moving_jaw_contact", "jaw")):
            setattr(self.scene, name, ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                                                     update_period=0., history_length=1, debug_vis=False,
                                                     filter_prim_paths_expr=filters))
        program_path = folder / "task_program.json"
        if program_path.is_file():
            from ..program import read_program

            self.task_program = read_program(program_path, spec)
            self.observations.policy.task_program = ObservationTermCfg(func=program_progress)
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
