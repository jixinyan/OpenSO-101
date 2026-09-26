# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from openso101.teleop.so101_mapping import batched_action_to_motor_units

from .config import digest

CAMERA_SIZE = 64


def student_features(proprio, wrist, overhead):
    images = [F.interpolate(image.float(), size=(CAMERA_SIZE, CAMERA_SIZE), mode="bilinear", align_corners=False)
              for image in (wrist, overhead)]
    return torch.cat((proprio / 100., *(image.flatten(1) for image in images)), dim=-1)


class VisionStudent(nn.Module):
    def __init__(self, num_actions=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(262, 128), nn.ELU(), nn.Linear(128, num_actions))

    def forward(self, features):
        proprio = features[:, :6]
        images = features[:, 6:].reshape(-1, 6, CAMERA_SIZE, CAMERA_SIZE)
        return self.head(torch.cat((proprio, self.encoder(images)), dim=-1))


class RLStudentPolicy:
    def __init__(self, folder: Path, device: str):
        self.metadata = json.loads((folder / "student.json").read_text())
        if self.metadata["schema_version"] != 1:
            raise ValueError("不支持的 student schema_version")
        for name, expected in self.metadata["files"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
                raise ValueError(f"student 文件校验失败：{name}")
        self.model = VisionStudent().to(device)
        self.model.load_state_dict(torch.load(folder / "student.pt", map_location=device, weights_only=True))
        self.model.eval()
        self.device = device
        self.openso101_preprocessor = self.preprocess
        self.openso101_postprocessor = self.decode_actions

    def preprocess(self, obs):
        return student_features(obs["observation.state"], obs["observation.images.wrist_camera"],
                                obs["observation.images.overhead_camera"])

    def select_action(self, features):
        return self.model(features)

    def decode_actions(self, actions):
        if actions.shape[-1] != 6 or not torch.isfinite(actions).all():
            raise ValueError("student action 必须包含六个有限数值")
        targets = []
        for item in self.metadata["action_mapping"]:
            value = actions[:, item["action_index"]]
            if item["type"] == "position":
                value = value * item["scale"] + item["offset"]
            elif item["type"] == "binary":
                value = torch.where(value < 0, item["close"], item["open"])
            else:
                raise ValueError("未知 student action 类型")
            targets.append(value.clamp(item["lower"], item["upper"]))
        return batched_action_to_motor_units(torch.stack(targets, dim=-1))
