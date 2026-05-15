# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task-local MDP reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from openso101.robots import SO101_GRIPPER_JOINT_NAMES
from openso101.tasks.shared.rewards import gripper_joint_pos, object_controlled_by_gripper, object_is_static

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_cube_top_was_lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-env episode flag for a controlled cube-top lift."""
    if not hasattr(env, "_cube_top_was_lifted"):
        env._cube_top_was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._cube_top_was_lifted[env.episode_length_buf <= 1] = False
    return env._cube_top_was_lifted


def ee_to_cube_top_distance(
    env: ManagerBasedRLEnv,
    std: float,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for moving the end-effector close to cube_top (tanh kernel)."""
    cube_top: RigidObject = env.scene[cube_top_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(cube_top.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    return 1.0 - torch.tanh(dist / std)


def cube_top_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    grasp_distance_threshold: float = 0.07,
    closed_threshold: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Binary reward when cube_top is lifted while controlled by the gripper."""
    lifted = object_controlled_by_gripper(
        env,
        minimal_height,
        grasp_distance_threshold,
        closed_threshold,
        robot_cfg,
        cube_top_cfg,
        ee_frame_cfg,
        gripper_cfg,
    )
    was_lifted = get_cube_top_was_lifted(env)
    was_lifted |= lifted
    env._cube_top_was_lifted = was_lifted
    return lifted.float()


def cubes_xy_aligned(
    env: ManagerBasedRLEnv,
    std: float,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_bottom"),
) -> torch.Tensor:
    """Reward for aligning cube_top over cube_bottom in the xy-plane (tanh kernel)."""
    cube_top: RigidObject = env.scene[cube_top_cfg.name]
    cube_bottom: RigidObject = env.scene[cube_bottom_cfg.name]
    xy_dist = torch.norm(
        cube_top.data.root_pos_w[:, :2] - cube_bottom.data.root_pos_w[:, :2], dim=1
    )
    elevated = (cube_top.data.root_pos_w[:, 2] > 0.025).float()
    was_lifted = get_cube_top_was_lifted(env).float()
    return was_lifted * elevated * (1.0 - torch.tanh(xy_dist / std))


def cubes_stacked(
    env: ManagerBasedRLEnv,
    std: float,
    cube_height: float = 0.03,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_bottom"),
) -> torch.Tensor:
    """Reward when cube_top is resting stably on top of cube_bottom."""
    cube_top: RigidObject = env.scene[cube_top_cfg.name]
    cube_bottom: RigidObject = env.scene[cube_bottom_cfg.name]

    target = cube_bottom.data.root_pos_w[:, :3].clone()
    target[:, 2] = target[:, 2] + cube_height

    dist = torch.norm(cube_top.data.root_pos_w[:, :3] - target, dim=1)
    elevated = (cube_top.data.root_pos_w[:, 2] > (cube_bottom.data.root_pos_w[:, 2] + cube_height * 0.5)).float()
    was_lifted = get_cube_top_was_lifted(env).float()
    return was_lifted * elevated * (1.0 - torch.tanh(dist / std))


def cube_released_on_stack(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    height_tolerance: float,
    cube_height: float = 0.03,
    open_threshold: float = 0.35,
    object_static_threshold: float = 0.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_top"),
    cube_bottom_cfg: SceneEntityCfg = SceneEntityCfg("cube_bottom"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Terminal-style reward for releasing a settled cube_top on cube_bottom."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube_top: RigidObject = env.scene[cube_top_cfg.name]
    cube_bottom: RigidObject = env.scene[cube_bottom_cfg.name]

    xy_dist = torch.norm(cube_top.data.root_pos_w[:, :2] - cube_bottom.data.root_pos_w[:, :2], dim=1)
    target_z = cube_bottom.data.root_pos_w[:, 2] + cube_height
    height_error = torch.abs(cube_top.data.root_pos_w[:, 2] - target_z)

    stacked = (xy_dist < xy_threshold) & (height_error < height_tolerance)
    was_lifted = get_cube_top_was_lifted(env)
    gripper_open = gripper_joint_pos(robot, gripper_cfg) > open_threshold
    top_static = object_is_static(env, object_static_threshold, cube_top_cfg)
    bottom_static = object_is_static(env, object_static_threshold, cube_bottom_cfg)
    return (was_lifted & stacked & gripper_open & top_static & bottom_static).float()


__all__ = [
    "get_cube_top_was_lifted",
    "ee_to_cube_top_distance",
    "cube_top_is_lifted",
    "cubes_xy_aligned",
    "cubes_stacked",
    "cube_released_on_stack",
]
