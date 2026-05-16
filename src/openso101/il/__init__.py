# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Imitation-learning pillar — thin functional wrappers over LeRobot.

The submodules under `policies/`, `runners/`, and `datasets/` are real,
importable APIs (not skeletons). They delegate to LeRobot's maintained
ACT/Diffusion/training stack so callers get every upstream improvement
automatically, while keeping a stable `openso101.il.*` namespace.

Typical use::

    from openso101.il.datasets import load_lerobot_dataset
    from openso101.il.runners import train_il_policy
    from openso101.il.policies import load_policy

    ds = load_lerobot_dataset("teleop_data/openso101_pickplace")
    result = train_il_policy(policy="act", dataset=ds.root, steps=200_000)
    policy = load_policy(result.last_checkpoint, device="cuda")
"""

from .datasets import LeRobotDatasetHandle, load_lerobot_dataset, summarize_lerobot_dataset
from .policies import (
    ACTPolicy,
    DiffusionPolicy,
    load_act_policy,
    load_diffusion_policy,
    load_policy,
    policy_class,
)
from .runners import TrainResult, train_il_policy

__all__ = [
    "ACTPolicy",
    "DiffusionPolicy",
    "LeRobotDatasetHandle",
    "TrainResult",
    "load_act_policy",
    "load_diffusion_policy",
    "load_lerobot_dataset",
    "load_policy",
    "policy_class",
    "summarize_lerobot_dataset",
    "train_il_policy",
]
