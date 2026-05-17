# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task-local MDP termination functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lift_success_height_only(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    goal_radius: float,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Episode ends when (object above ``minimal_height``) AND
    (object within ``goal_radius`` of command target).

    No grasp / gripper-closed / static requirements. Mirrors the upstream
    MuammerBay/isaac_so_arm101 success criterion: a height-only gate plus
    proximity in the robot-root command frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) & (distance < goal_radius)


__all__ = ["lift_success_height_only"]
