# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task-local MDP reward functions.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/rewards.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward the agent for lifting the object above the minimal height."""
    raise NotImplementedError(
        "object_is_lifted not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/rewards.py"
    )


def object_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward the agent for reaching the object using tanh-kernel."""
    raise NotImplementedError(
        "object_ee_distance not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/rewards.py"
    )


def object_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    raise NotImplementedError(
        "object_goal_distance not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/rewards.py"
    )


def object_goal_distance_controlled(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    grasp_distance_threshold: float,
    closed_threshold: float,
    command_name: str,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward goal tracking only while the object is controlled by the gripper."""
    raise NotImplementedError(
        "object_goal_distance_controlled not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/rewards.py"
    )


def object_ee_distance_and_lifted(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Combined reward for reaching the object AND lifting it."""
    raise NotImplementedError(
        "object_ee_distance_and_lifted not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/rewards.py"
    )


__all__ = [
    "object_is_lifted",
    "object_ee_distance",
    "object_goal_distance",
    "object_goal_distance_controlled",
    "object_ee_distance_and_lifted",
]
