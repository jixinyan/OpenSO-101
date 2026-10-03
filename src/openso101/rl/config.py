# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    backend: Literal["rsl_rl", "sb3", "skrl", "rl_games"] = "rsl_rl"
    algo: Literal["ppo", "sac", "tqc"] = "ppo"
    seed: int = Field(default=42, ge=0)
    iterations: int = Field(default=2000, gt=0)
    rollout_steps: int = Field(default=96, gt=1)
    epochs: int = Field(default=5, gt=0)
    mini_batches: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip: float = Field(default=0.2, gt=0, lt=1)
    entropy_coef: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    normalize_observations: bool = True
    replay_size: int = Field(default=100000, gt=0)
    learning_starts: int = Field(default=1000, ge=0)
    environment_mode: Literal["randomized", "nominal"] = "randomized"
    action_distribution: Literal["gaussian", "tanh_gaussian"] = "gaussian"
    initial_noise_std: float = Field(default=0.5, gt=0)
    learning_rate_schedule: Literal["fixed", "adaptive"] = "fixed"
    desired_kl: float = Field(default=0.01, gt=0)
    evaluation_interval: int = Field(default=100, gt=0)
    evaluation_episodes: int = Field(default=100, gt=0)
    replay_batch_size: int = Field(default=256, gt=1)
    gradient_steps: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def supported_algorithm(self):
        if self.algo != "ppo" and self.backend != "sb3":
            raise ValueError("SAC 和 TQC 使用 sb3 backend")
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims 必须包含正整数")
        if self.action_distribution == "tanh_gaussian" and (self.backend != "rsl_rl" or self.algo != "ppo"):
            raise ValueError("tanh_gaussian 使用 rsl_rl PPO")
        if self.learning_rate_schedule == "adaptive" and self.backend != "rsl_rl":
            raise ValueError("adaptive learning rate schedule 使用 rsl_rl backend")
        if (self.backend == "skrl" or self.algo != "ppo") and not math.exp(-20) <= self.initial_noise_std <= math.exp(2):
            raise ValueError("当前策略的 initial_noise_std 需要位于 [exp(-20), exp(2)]")
        if self.action_distribution == "tanh_gaussian" and not math.exp(-5) <= self.initial_noise_std <= math.exp(2):
            raise ValueError("tanh_gaussian 的 initial_noise_std 需要位于 [exp(-5), exp(2)]")
        if self.learning_starts >= self.replay_size:
            raise ValueError("learning_starts 必须小于 replay_size")
        return self

    def batch_size(self, num_envs: int) -> int:
        total = self.rollout_steps * num_envs
        if num_envs <= 0 or total % self.mini_batches or total // self.mini_batches < 2:
            raise ValueError("rollout_steps × num_envs 必须能够整除 mini_batches，且每组至少包含两个样本")
        return total // self.mini_batches


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CheckpointMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    task_id: str
    task_profile: Literal["default", "grasp_v2", "grasp_v3", "grasp_v4"] = "default"
    config: TrainCfg
    observation_format: str = "state"
    git_sha: str
    checkpoint: str
    files: dict[str, str]
    scene_sha256: str | None = None
    completed_transitions: int = Field(ge=0)

    def write(self, folder: Path):
        (folder / "checkpoint.json").write_text(self.model_dump_json(indent=2))

    @classmethod
    def read(cls, folder: Path):
        meta = cls.model_validate_json((folder / "checkpoint.json").read_text())
        root = folder.resolve()
        if meta.checkpoint not in meta.files:
            raise ValueError("checkpoint 缺少文件校验信息")
        for relative, expected in meta.files.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or digest(path) != expected:
                raise ValueError(f"checkpoint 文件验证失败：{relative}")
        return meta


def write_backend_config(folder: Path, config: dict):
    (folder / "backend.json").write_text(json.dumps(config, indent=2))
