import json
import math
from pathlib import Path

import torch

from .config import digest


class PortablePolicy:
    def __init__(self, folder: Path, device: str = "cpu"):
        folder = Path(folder).resolve()
        self.metadata = json.loads((folder / "policy.json").read_text())
        if self.metadata["schema_version"] != 1:
            raise ValueError("不支持的 portable policy schema_version")
        for name, expected in self.metadata["files"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or digest(path) != expected:
                raise ValueError(f"policy 文件校验失败：{name}")
        self.device = torch.device(device)
        self.model = torch.jit.load(str(folder / "policy.pt"), map_location=self.device).eval()
        self.observation_dim = sum(term["size"] for term in self.metadata["observation_terms"])
        if not math.isfinite(self.metadata["control_dt"]) or self.metadata["control_dt"] <= 0:
            raise ValueError("control_dt 必须为正数")

    def observation(self, joint_position, joint_velocity, object_position_root, goal_root, grasp_state, last_action):
        joint_position = torch.as_tensor(joint_position, device=self.device, dtype=torch.float32)
        if joint_position.ndim != 2 or joint_position.shape[1] != len(self.metadata["observation_joint_names"]):
            raise ValueError("joint_position 形状不匹配")
        defaults = torch.tensor(self.metadata["default_joint_positions"], device=self.device)
        velocity_defaults = torch.tensor(self.metadata["default_joint_velocities"], device=self.device)
        values = {
            "joint_pos": joint_position - defaults,
            "joint_vel": torch.as_tensor(joint_velocity, device=self.device, dtype=torch.float32) - velocity_defaults,
            "object_position": torch.as_tensor(object_position_root, device=self.device, dtype=torch.float32),
            "target_object_position": torch.as_tensor(goal_root, device=self.device, dtype=torch.float32),
            "grasp_state": torch.as_tensor(grasp_state, device=self.device, dtype=torch.float32),
            "actions": torch.as_tensor(last_action, device=self.device, dtype=torch.float32),
        }
        batch = joint_position.shape[0]
        ordered = []
        for term in self.metadata["observation_terms"]:
            value = values[term["name"]]
            if value.shape != (batch, term["size"]):
                raise ValueError(f"观测形状不匹配：{term['name']}")
            ordered.append(value)
        observation = torch.cat(ordered, dim=-1)
        if not torch.isfinite(observation).all():
            raise ValueError("观测必须包含有限数值")
        return observation

    def predict(self, observation):
        observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim or not torch.isfinite(observation).all():
            raise ValueError("policy 观测形状或数值无效")
        with torch.inference_mode():
            actions = self.model(observation)
        if actions.shape != (observation.shape[0], 6) or not torch.isfinite(actions).all():
            raise ValueError("policy 输出必须包含六个有限数值")
        return actions

    def joint_targets(self, actions, *, enforce_limits=True):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != 6 or not torch.isfinite(actions).all():
            raise ValueError("action 必须包含六个有限数值")
        targets = []
        for item in self.metadata["action_mapping"]:
            value = actions[:, item["action_index"]]
            if item["type"] == "position":
                value = value * item["scale"] + item["offset"]
            elif item["type"] == "binary":
                value = torch.where(value < 0, item["close"], item["open"])
            else:
                raise ValueError("不支持的 action 类型")
            if enforce_limits:
                value = value.clamp(item["lower"], item["upper"])
            targets.append(value)
        return torch.stack(targets, dim=-1)
