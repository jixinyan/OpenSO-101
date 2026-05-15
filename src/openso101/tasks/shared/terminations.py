# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared MDP termination functions used by built-in manipulation tasks.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/shared/terminations.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def object_left_gripper(
    env: "ManagerBasedRLEnv",
    max_distance: float = 0.15,
    only_when_above_height: float = 0.0,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Terminate (as failure) when the cube has escaped the gripper."""
    raise NotImplementedError(
        "object_left_gripper not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/terminations.py"
    )


__all__ = [
    "object_left_gripper",
]
