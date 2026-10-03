# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Termination predicates for the curriculum pick-and-place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
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


def released_at_place_goal(env, command_name="object_pose", settle_seconds=0.5):
    return (env.command_manager.get_term(command_name).placement_hold_seconds >= settle_seconds) & placement_eligible(env, command_name)


def placement_eligible(env, command_name="object_pose"):
    robot, obj = env.scene["robot"], env.scene["object"]
    command = env.command_manager.get_term(command_name)
    position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
    jaw = robot.data.joint_pos[:, robot.joint_names.index("Jaw")]
    placed = torch.linalg.vector_norm(position - command.goal_for_stage(2), dim=-1) <= .03
    stable = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1) <= .02
    stable &= torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=-1) <= .1
    return (command.stage == 2) & placed & stable & ~object_grasped_by_jaws(env, .5) & (jaw > .4)


class StablePlacementSuccess(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.hold_seconds = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.hold_seconds[env_ids if env_ids is not None else slice(None)] = 0.

    def __call__(self, env, command_name="object_pose", settle_seconds=.5):
        eligible = placement_eligible(env, command_name)
        self.hold_seconds.copy_(torch.where(eligible, self.hold_seconds + env.step_dt, 0.))
        env.command_manager.get_term(command_name).placement_hold_seconds = self.hold_seconds
        return self.hold_seconds >= settle_seconds


__all__ = ["reached_goal_while_grasped", "released_at_place_goal"]
