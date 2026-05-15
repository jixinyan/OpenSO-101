# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task-local MDP termination functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from openso101.robots import SO101_GRIPPER_JOINT_NAMES
from openso101.tasks.shared.rewards import object_controlled_by_gripper, object_is_static

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    minimal_height: float = 0.025,
    grasp_distance_threshold: float = 0.07,
    closed_threshold: float = 0.12,
    object_static_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Terminate when the held object reaches the goal position and settles."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    controlled = object_controlled_by_gripper(
        env,
        minimal_height,
        grasp_distance_threshold,
        closed_threshold,
        robot_cfg,
        object_cfg,
        ee_frame_cfg,
        gripper_cfg,
    )
    settled = object_is_static(env, object_static_threshold, object_cfg)
    return (distance < threshold) & controlled & settled


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
    proximity in the robot-root command frame. Frame transform matches
    ``object_reached_goal`` (above) for consistency.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) & (distance < goal_radius)


__all__ = ["object_reached_goal", "lift_success_height_only"]
