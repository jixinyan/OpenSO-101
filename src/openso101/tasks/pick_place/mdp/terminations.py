# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Termination predicates for the curriculum pick-and-place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from openso101.tasks.shared.grasp import object_grasped_by_jaws

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reached_goal_while_grasped(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    threshold: float = 0.03,
    force_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sentinel ``InGoalRegion`` success: cube in the goal sphere AND held.

    Success requires BOTH the cube surface touching the (air) goal sphere and
    a contact-confirmed grasp. The grasp gate is what makes this a genuine
    pick-and-lift success rather than a launch-and-fly exploit: a cube that
    drifts through the goal region after being swatted is not held, so it does
    not count.

    The goal location is whatever the command term currently exposes (frozen
    by ``lock_stage`` for this task), so this predicate is independent of any
    curriculum staging.
    """
    cmd_term = env.command_manager.get_term(command_name)

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
    )
    in_goal = cmd_term.is_touching_goal(cube_pos_b, threshold=threshold)
    grasped = object_grasped_by_jaws(env, force_threshold)
    return in_goal & grasped


__all__ = ["reached_goal_while_grasped"]
