# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared MDP reward functions used by built-in manipulation tasks.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def joint_pos_delta_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Penalise the L2 norm of the per-step joint-position change."""
    raise NotImplementedError(
        "joint_pos_delta_l2 not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


def gripper_joint_pos(
    robot: "Articulation",
    gripper_cfg: "SceneEntityCfg",
) -> "torch.Tensor":
    """Return the scalar gripper joint position for each environment."""
    raise NotImplementedError(
        "gripper_joint_pos not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


def rigid_object_speed(obj: "RigidObject") -> "torch.Tensor":
    """Return combined linear/angular object speed."""
    raise NotImplementedError(
        "rigid_object_speed not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


def object_is_static(
    env: "ManagerBasedRLEnv",
    object_static_threshold: float,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Return true when a rigid object is nearly settled."""
    raise NotImplementedError(
        "object_is_static not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


def close_gripper_near_object(
    env: "ManagerBasedRLEnv",
    near_threshold: float,
    closed_std: float,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward closing the gripper only when the end-effector is near the object."""
    raise NotImplementedError(
        "close_gripper_near_object not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


def object_controlled_by_gripper(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    grasp_distance_threshold: float,
    closed_threshold: float,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Return true when an object is elevated, close to the EE, and the gripper is closed."""
    raise NotImplementedError(
        "object_controlled_by_gripper not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


def object_controlled_by_gripper_reward(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    grasp_distance_threshold: float,
    closed_threshold: float,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Binary reward for a lifted object still plausibly controlled by the gripper."""
    raise NotImplementedError(
        "object_controlled_by_gripper_reward not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/rewards.py"
    )


__all__ = [
    "joint_pos_delta_l2",
    "gripper_joint_pos",
    "rigid_object_speed",
    "object_is_static",
    "close_gripper_near_object",
    "object_controlled_by_gripper",
    "object_controlled_by_gripper_reward",
]
