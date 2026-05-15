# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Reward functions for the curriculum-driven pick-and-place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_to_curriculum_stage_goal(
    env: "ManagerBasedRLEnv",
    stage: int,
    std: float,
    minimal_height: float | None = 0.04,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Tanh-kernel reward to one stage goal, active only during that stage.

    This keeps each phase diagnosable in TensorBoard while still using the same
    moving green ball for the policy's observation. Stage 0/1 can be height-gated
    so a cube resting below the ball does not collect the goal-tracking reward.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
    )
    goal_b = cmd_term.goal_for_stage(stage)
    distance = torch.norm(cube_pos_b - goal_b, dim=1)
    reward = 1.0 - torch.tanh(distance / std)

    active_stage = cmd_term.stage == stage
    if minimal_height is not None:
        active_stage = active_stage & (cube_pos_b[:, 2] > minimal_height)
    return active_stage.float() * reward


def stage_completion_bonus(
    env: "ManagerBasedRLEnv",
    completed_stage: int,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """One-step sparse reward when touching a stage goal advances the command."""
    cmd_term = env.command_manager.get_term(command_name)
    return (cmd_term.just_completed_stage == completed_stage).float()


def object_is_lifted_when_grasped(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    force_threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """``object_is_lifted`` gated on contact-confirmed grasp.

    Returns 1.0 only when the cube is above ``minimal_height`` (world frame) AND
    both SO-101 jaws are in contact with it. Defeats the failure mode where a
    policy knocks the cube off the table to satisfy a pure-height check.
    """
    from openso101.tasks.pick_place.mdp.grasp import object_grasped_by_jaws

    obj: RigidObject = env.scene[object_cfg.name]
    height_ok = obj.data.root_pos_w[:, 2] > minimal_height
    grasped = object_grasped_by_jaws(env, force_threshold)
    return (height_ok & grasped).float()


def cube_to_curriculum_stage_goal_when_grasped(
    env: "ManagerBasedRLEnv",
    stage: int,
    std: float,
    minimal_height: float | None = 0.04,
    force_threshold: float = 0.5,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """:func:`cube_to_curriculum_stage_goal` multiplicatively gated on grasp.

    Stage 0 (lift) and stage 1 (carry) require the cube to be held to count;
    stage 2 (place) does NOT gate on grasp -- the policy must release the cube
    to win, so requiring continuous grasp at placement would be wrong.
    Callers should still use the un-gated ``cube_to_curriculum_stage_goal`` for
    stage 2; this function returns the un-gated value if called with stage=2.

    Retained for backward compatibility; the active reward chain uses
    :func:`cube_to_curriculum_stage_goal_height_gated` instead so the policy
    can bootstrap before contact sensors fire.
    """
    from openso101.tasks.pick_place.mdp.grasp import object_grasped_by_jaws

    base = cube_to_curriculum_stage_goal(
        env, stage, std, minimal_height, command_name, robot_cfg, object_cfg,
    )
    if stage == 2:
        return base
    grasped = object_grasped_by_jaws(env, force_threshold).float()
    return base * grasped


def cube_to_curriculum_stage_goal_height_gated(
    env: "ManagerBasedRLEnv",
    stage: int,
    std: float,
    minimal_height: float,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Stage-specific goal-distance reward, gated only on object height.

    Replaces the contact-confirmed grasp gate in
    :func:`cube_to_curriculum_stage_goal_when_grasped` with a height-only gate
    matching the upstream lift task's pattern. Once the cube rises above
    ``minimal_height``, the tanh-shaped goal-tracking reward is awarded.
    Only fires for envs whose current curriculum stage equals ``stage``; all
    other envs/stages get zero, preserving the staged learning structure.

    Mirrors the frame computation of :func:`cube_to_curriculum_stage_goal`
    exactly (robot-root-frame distance, ``cmd_term.goal_for_stage``); the
    only difference is the gate uses object height instead of jaw contact.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
    )
    goal_b = cmd_term.goal_for_stage(stage)
    distance = torch.norm(cube_pos_b - goal_b, dim=1)
    reward = 1.0 - torch.tanh(distance / std)

    active_stage = cmd_term.stage == stage
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return (active_stage & lifted).float() * reward


__all__ = [
    "cube_to_curriculum_stage_goal",
    "stage_completion_bonus",
    "object_is_lifted_when_grasped",
    "cube_to_curriculum_stage_goal_when_grasped",
    "cube_to_curriculum_stage_goal_height_gated",
]
