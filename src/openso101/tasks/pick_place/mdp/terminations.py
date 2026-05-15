# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Termination predicates for the curriculum pick-and-place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def curriculum_complete(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Episode succeeds when final-stage cube surface touches the goal sphere.

    Reads the per-env stage tensor from the curriculum command term, so it cannot
    fire on stage 0 or stage 1 successes (those just advance the stage and the
    episode continues).
    """
    cmd_term = env.command_manager.get_term(command_name)
    at_final = cmd_term.stage == 2

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
    )
    return at_final & cmd_term.is_touching_goal(cube_pos_b, threshold=threshold)


__all__ = ["curriculum_complete"]
