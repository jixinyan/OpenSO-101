# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task-local MDP observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
) -> torch.Tensor:
    """Position of an object expressed in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj_pos_w
    )
    return obj_pos_b


def cube_top_to_cube_bottom_offset(
    env: "ManagerBasedRLEnv",
    cube_height: float = 0.03,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_bottom"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from cube_top to the stacking goal (top of cube_bottom) in robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube_top: RigidObject = env.scene[cube_top_cfg.name]
    cube_bottom: RigidObject = env.scene[cube_bottom_cfg.name]

    target_w = cube_bottom.data.root_pos_w[:, :3].clone()
    target_w[:, 2] = target_w[:, 2] + cube_height

    top_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], cube_top.data.root_pos_w[:, :3]
    )
    goal_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_w
    )
    return goal_pos_b - top_pos_b


__all__ = [
    "object_position_in_robot_root_frame",
    "cube_top_to_cube_bottom_offset",
]
