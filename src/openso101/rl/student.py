# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from openso101.teleop.so101_mapping import batched_action_to_motor_units, batched_motor_units_to_action

from .config import digest
from .portable import decode_joint_targets

CAMERA_SIZE = 64


def student_features(proprio, wrist, overhead, goal=None):
    images = [F.interpolate(image.float(), size=(CAMERA_SIZE, CAMERA_SIZE), mode="bilinear", align_corners=False)
              for image in (wrist, overhead)]
    task = () if goal is None else (goal,)
    if goal is not None and (goal.shape != (proprio.shape[0], 3) or not torch.isfinite(goal).all()):
        raise ValueError("student 目标需要每个环境的三个米制坐标")
    return torch.cat((proprio / 100., *task, *(image.flatten(1) for image in images)), dim=-1)


def student_goal(env):
    if hasattr(env.cfg, "scene_spec"):
        from isaaclab.utils.math import subtract_frame_transforms

        robot = env.scene["robot"]
        position = torch.tensor(env.cfg.scene_spec.task.goal_position_m, device=env.device).expand(env.num_envs, -1)
        goal, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                            position + env.scene.env_origins)
    else:
        goal = env.command_manager.get_command("object_pose")[:, :3]
    if goal.shape != (env.num_envs, 3) or not torch.isfinite(goal).all():
        raise ValueError("student 需要有效的 robot root frame 任务目标")
    return goal


class VisionStudent(nn.Module):
    def __init__(self, num_actions=6, bounded_actions=False, goal_dim=0):
        super().__init__()
        if goal_dim not in (0, 3):
            raise ValueError("student goal_dim 必须为 0 或 3")
        self.goal_dim = goal_dim
        self.bounded_actions = bounded_actions
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(262 + goal_dim, 128), nn.ELU(), nn.Linear(128, num_actions))

    def forward(self, features):
        if features.shape[1] != 6 + self.goal_dim + 6 * CAMERA_SIZE ** 2 or not torch.isfinite(features).all():
            raise ValueError("student 输入形状或数值无效")
        proprio = features[:, :6 + self.goal_dim]
        images = features[:, 6 + self.goal_dim:].reshape(-1, 6, CAMERA_SIZE, CAMERA_SIZE)
        actions = self.head(torch.cat((proprio, self.encoder(images)), dim=-1))
        if not torch.isfinite(actions).all():
            raise RuntimeError("student 产生无效动作")
        return actions.tanh() if self.bounded_actions else actions


class RLStudentPolicy:
    def __init__(self, folder: Path, device: str):
        self.metadata = json.loads((folder / "student.json").read_text())
        if self.metadata["schema_version"] not in (1, 2):
            raise ValueError("不支持的 student schema_version")
        if self.metadata["schema_version"] == 2 and self.metadata["goal_input"] != "robot_root_xyz_m":
            raise ValueError("student 目标坐标格式无效")
        for name, expected in self.metadata["files"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
                raise ValueError(f"student 文件校验失败：{name}")
        self.model = VisionStudent(bounded_actions=self.metadata.get("bounded_actions", False),
                                   goal_dim=3 if self.metadata["schema_version"] == 2 else 0).to(device)
        self.model.load_state_dict(torch.load(folder / "student.pt", map_location=device, weights_only=True))
        self.model.eval()
        self.device = device
        self.openso101_preprocessor = self.preprocess
        self.openso101_postprocessor = self.decode_actions

    def preprocess(self, obs):
        self.joint_position = batched_motor_units_to_action(obs["observation.state"])
        goal = obs["observation.goal"] if self.model.goal_dim else None
        return student_features(obs["observation.state"], obs["observation.images.wrist_camera"],
                                obs["observation.images.overhead_camera"], goal)

    def select_action(self, features):
        return self.model(features)

    def decode_actions(self, actions):
        targets = decode_joint_targets(actions, self.metadata["action_mapping"],
                                       joint_position=self.joint_position)
        return batched_action_to_motor_units(targets)
