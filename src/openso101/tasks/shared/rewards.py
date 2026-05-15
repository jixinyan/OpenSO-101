# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared reward functions for all tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from openso101.robots import SO101_GRIPPER_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_PREV_JOINT_POS_KEY = "_prev_joint_pos_smoothness"


def joint_pos_delta_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the L2 norm of the per-step joint-position change.

    Stores the previous step's joint positions in ``env.extras`` and returns
    ``||q_t - q_{t-1}||^2`` per env. Reduces high-frequency jitter in the
    learned policy without requiring access to joint velocities.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or (isinstance(joint_ids, slice) and joint_ids == slice(None)):
        joint_pos = asset.data.joint_pos
    else:
        joint_pos = asset.data.joint_pos[:, joint_ids]

    prev = env.extras.get(_PREV_JOINT_POS_KEY)
    if prev is None or prev.shape != joint_pos.shape:
        prev = joint_pos.clone()

    delta = joint_pos - prev
    # Zero the delta for envs that just reset (the teleport would register as a
    # huge single-step jump).  Thresholded per-env, not per-joint.
    max_step_delta = 0.3  # rad; well above any realistic single-step motion
    is_reset = (delta.abs() > max_step_delta).any(dim=1, keepdim=True)
    delta = torch.where(is_reset.expand_as(delta), torch.zeros_like(delta), delta)
    penalty = torch.sum(delta * delta, dim=1)

    env.extras[_PREV_JOINT_POS_KEY] = joint_pos.detach().clone()
    return penalty


def gripper_joint_pos(
    robot: Articulation,
    gripper_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the scalar gripper joint position for each environment."""
    joint_ids = gripper_cfg.joint_ids
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(gripper_cfg.joint_names)

    joint_pos = robot.data.joint_pos[:, joint_ids]
    if joint_pos.ndim == 1:
        return joint_pos
    return joint_pos.mean(dim=1)


def rigid_object_speed(obj: RigidObject) -> torch.Tensor:
    """Return combined linear/angular object speed."""
    linear_speed = torch.norm(obj.data.root_state_w[:, 7:10], dim=1)
    angular_speed = torch.norm(obj.data.root_state_w[:, 10:13], dim=1)
    return linear_speed + angular_speed


def object_is_static(
    env: ManagerBasedRLEnv,
    object_static_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return true when a rigid object is nearly settled."""
    obj: RigidObject = env.scene[object_cfg.name]
    return rigid_object_speed(obj) < object_static_threshold


def close_gripper_near_object(
    env: ManagerBasedRLEnv,
    near_threshold: float,
    closed_std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Reward closing the gripper only when the end-effector is near the object."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    near_object = (ee_dist < near_threshold).float()
    gripper_pos = gripper_joint_pos(robot, gripper_cfg).clamp(min=0.0)
    close_reward = 1.0 - torch.tanh(gripper_pos / closed_std)
    return near_object * close_reward


def object_controlled_by_gripper(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    grasp_distance_threshold: float,
    closed_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Return true when an object is elevated, close to the EE, and the gripper is closed."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    grip_pos = gripper_joint_pos(robot, gripper_cfg)
    return (
        (obj.data.root_pos_w[:, 2] > minimal_height)
        & (ee_dist < grasp_distance_threshold)
        & (grip_pos < closed_threshold)
    )


def object_controlled_by_gripper_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    grasp_distance_threshold: float,
    closed_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES)),
) -> torch.Tensor:
    """Binary reward for a lifted object still plausibly controlled by the gripper."""
    return object_controlled_by_gripper(
        env,
        minimal_height,
        grasp_distance_threshold,
        closed_threshold,
        robot_cfg,
        object_cfg,
        ee_frame_cfg,
        gripper_cfg,
    ).float()


__all__ = [
    "joint_pos_delta_l2",
    "gripper_joint_pos",
    "rigid_object_speed",
    "object_is_static",
    "close_gripper_near_object",
    "object_controlled_by_gripper",
    "object_controlled_by_gripper_reward",
]
