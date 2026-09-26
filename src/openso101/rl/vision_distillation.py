# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import copy
import json
from pathlib import Path

import torch
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms import Distillation
from rsl_rl.modules import StudentTeacher
from rsl_rl.runners import DistillationRunner

from openso101.robots.so101.constants import SO101_SIM_JOINT_NAMES
from openso101.teleop.so101_mapping import batched_action_to_motor_units

from .config import CheckpointMeta, digest
from .student import VisionStudent, student_features


def camera_proprio_observation(env):
    robot = env.scene["robot"]
    ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
    proprio = batched_action_to_motor_units(robot.data.joint_pos[:, ids])
    images = [env.scene[name].data.output["rgb"][:, :, :, :3].permute(0, 3, 1, 2).float() / 255.
              for name in ("wrist_camera", "overhead_camera")]
    return student_features(proprio, *images)


@configclass
class StudentObservationsCfg(ObservationGroupCfg):
    features = ObservationTermCfg(func=camera_proprio_observation)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


class VisionStudentTeacher(StudentTeacher):
    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        # 构造过程中使用本体观测确定 student 输入，图像由 VisionStudent 编码。
        proprio_obs = {**obs, "student": obs["student"][:, :6]}
        super().__init__(proprio_obs, obs_groups, num_actions, student_obs_normalization=False, **kwargs)
        self.student = VisionStudent(num_actions)
        self.teacher.requires_grad_(False)
        self.teacher_obs_normalizer.requires_grad_(False)


class VisionDistillationRunner(DistillationRunner):
    def _construct_algorithm(self, obs):
        policy = VisionStudentTeacher(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)
        algorithm = Distillation(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        algorithm.init_storage("distillation", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm


def action_mapping(env):
    from isaaclab.envs.mdp.actions import BinaryJointPositionAction, JointPositionAction

    robot = env.scene["robot"]
    mappings = {}
    action_index = 0
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        ids = list(range(robot.num_joints)) if isinstance(term._joint_ids, slice) else list(term._joint_ids)
        for index, joint_id in enumerate(ids):
            joint_name = robot.joint_names[joint_id]
            lower, upper = robot.data.soft_joint_pos_limits[0, joint_id].tolist()
            item = {"joint_name": joint_name, "action_index": action_index + index, "lower": lower, "upper": upper}
            if isinstance(term, JointPositionAction):
                scale = torch.as_tensor(term._scale).expand(env.num_envs, len(ids))[0, index]
                offset = torch.as_tensor(term._offset).expand(env.num_envs, len(ids))[0, index]
                item.update(type="position", scale=float(scale), offset=float(offset))
            elif type(term) is BinaryJointPositionAction:
                item.update(type="binary", action_index=action_index,
                            close=float(term._close_command[0, index]), open=float(term._open_command[0, index]))
            else:
                raise ValueError(f"student 导出不支持 action term：{type(term).__name__}")
            if joint_name in mappings:
                raise ValueError("多个 action term 控制同一关节")
            mappings[joint_name] = item
        action_index += term.action_dim
    if set(mappings) != set(SO101_SIM_JOINT_NAMES) or action_index != 6:
        raise ValueError("student 导出需要完整的六个 SO-101 关节")
    return [mappings[name] for name in SO101_SIM_JOINT_NAMES]


def train_vision_student(env, teacher_folder: Path, output: Path, iterations: int, rollout_steps: int):
    meta = CheckpointMeta.read(teacher_folder)
    if meta.config.backend != "rsl_rl" or meta.config.algo != "ppo":
        raise ValueError("视觉蒸馏需要 rsl_rl PPO teacher")
    policy_cfg = json.loads((teacher_folder / "backend.json").read_text())["policy"]
    cfg = {
        "num_steps_per_env": rollout_steps, "save_interval": iterations, "logger": "tensorboard",
        "obs_groups": {"policy": ["student"], "teacher": ["policy"]},
        "policy": {"teacher_hidden_dims": policy_cfg["actor_hidden_dims"],
                   "teacher_obs_normalization": policy_cfg["actor_obs_normalization"],
                   "activation": policy_cfg["activation"], "init_noise_std": 0.1},
        "algorithm": {"num_learning_epochs": 1, "learning_rate": 1e-4, "gradient_length": 1,
                      "max_grad_norm": 1., "loss_type": "mse"},
    }
    (output / "distillation.json").write_text(json.dumps(copy.deepcopy(cfg), indent=2))
    runner = VisionDistillationRunner(RslRlVecEnvWrapper(env), cfg, log_dir=str(output), device=env.unwrapped.device)
    runner.load(str(teacher_folder / meta.checkpoint), load_optimizer=False)
    runner.learn(iterations)
    runner.save(str(output / "distillation.pt"))
    torch.save(runner.alg.policy.student.state_dict(), output / "student.pt")
    (output / "student.json").write_text(json.dumps({
        "schema_version": 1, "task_id": meta.task_id, "teacher_sha256": meta.files[meta.checkpoint],
        "scene_sha256": meta.scene_sha256, "control_dt": env.unwrapped.step_dt,
        "observation_format": "motor_positions_and_two_rgb_cameras", "image_size": [64, 64],
        "action_mapping": action_mapping(env.unwrapped), "files": {"student.pt": digest(output / "student.pt")},
    }, indent=2))
