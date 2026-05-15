# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared termination conditions for atomic skills that start with the cube
already in the gripper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_left_gripper(
    env: ManagerBasedRLEnv,
    max_distance: float = 0.15,
    only_when_above_height: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Terminate (as failure) when the cube has escaped the gripper.

    Escape = end-effector–to-cube distance exceeds ``max_distance``.

    Args:
        only_when_above_height: If > 0, only consider the cube lost when it is
            still airborne.  Used by Place so the legitimate release onto the
            table surface does not trigger a false failure.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    lost = dist > max_distance
    if only_when_above_height > 0.0:
        airborne = obj.data.root_pos_w[:, 2] > only_when_above_height
        lost = lost & airborne
    return lost


__all__ = [
    "object_left_gripper",
]
