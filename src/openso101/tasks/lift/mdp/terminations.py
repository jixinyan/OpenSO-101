# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task-local MDP termination functions.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/terminations.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def object_reached_goal(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    threshold: float = 0.02,
    minimal_height: float = 0.025,
    grasp_distance_threshold: float = 0.07,
    closed_threshold: float = 0.12,
    object_static_threshold: float = 0.5,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Terminate when the held object reaches the goal position and settles."""
    raise NotImplementedError(
        "object_reached_goal not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/terminations.py"
    )


def lift_success_height_only(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    goal_radius: float,
    command_name: str = "object_pose",
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Episode ends when (object above ``minimal_height``) AND
    (object within ``goal_radius`` of command target).

    No grasp / gripper-closed / static requirements. Mirrors the upstream
    MuammerBay/isaac_so_arm101 success criterion: a height-only gate plus
    proximity in the robot-root command frame.
    """
    raise NotImplementedError(
        "lift_success_height_only not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/terminations.py"
    )


__all__ = ["object_reached_goal", "lift_success_height_only"]
