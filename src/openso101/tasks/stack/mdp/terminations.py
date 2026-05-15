# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task-local MDP termination functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from openso101.robots import SO101_GRIPPER_JOINT_NAMES
from openso101.tasks.shared.rewards import gripper_joint_pos, object_is_static

from .rewards import get_cube_top_was_lifted

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
) -> torch.Tensor:
    """Terminate when a cube falls below the table surface."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w[:, 2] < minimum_height


def cubes_stacked_success(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.015,
    height_tolerance: float = 0.01,
    cube_height: float = 0.03,
    open_threshold: float = 0.35,
    object_static_threshold: float = 0.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_bottom"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Terminate when cube_top was lifted, released, stacked, and both cubes settle."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube_top: RigidObject = env.scene[cube_top_cfg.name]
    cube_bottom: RigidObject = env.scene[cube_bottom_cfg.name]

    xy_dist = torch.norm(
        cube_top.data.root_pos_w[:, :2] - cube_bottom.data.root_pos_w[:, :2], dim=1
    )
    target_z = cube_bottom.data.root_pos_w[:, 2] + cube_height
    height_error = torch.abs(cube_top.data.root_pos_w[:, 2] - target_z)

    stacked = (xy_dist < xy_threshold) & (height_error < height_tolerance)
    was_lifted = get_cube_top_was_lifted(env)
    gripper_open = gripper_joint_pos(robot, gripper_cfg) > open_threshold
    top_static = object_is_static(env, object_static_threshold, cube_top_cfg)
    bottom_static = object_is_static(env, object_static_threshold, cube_bottom_cfg)

    return stacked & was_lifted & gripper_open & top_static & bottom_static


__all__ = [
    "cube_dropped",
    "cubes_stacked_success",
]
