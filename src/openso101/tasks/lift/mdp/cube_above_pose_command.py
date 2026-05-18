# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Goal-pose command for the lift task that locks the goal to ``(cube_xy,
cube_z + lift_height)`` at every resample.

The standard ``UniformPoseCommand`` samples the goal pose in the robot root
frame, independently of where the cube is. For a "lift to a sphere above the
cube" task, that produces goals that may be off to the side and gives a
misleading visualization. This subclass overrides ``_resample_command`` to
read the object's current world position (set by the cube-reset event a moment
earlier) and place the goal directly above it.

With ``resampling_time_range == episode_length_s`` (the lift config does
this), the resample fires exactly once per episode, immediately after the
reset event — so the goal is locked to the *initial* cube xy of each episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CubeAbovePoseCommand(UniformPoseCommand):
    """Goal pose at ``(cube_xy, cube_z + lift_height)``, identity orientation
    (or sampled within ``ranges.roll/pitch/yaw`` if non-zero).

    ``lift_height`` is sampled uniformly from ``cfg.lift_height_range`` on
    each resample so the policy sees a variety of lift heights during
    training.
    """

    cfg: "CubeAbovePoseCommandCfg"

    def __init__(self, cfg: "CubeAbovePoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.object: RigidObject = env.scene[cfg.object_name]

    def _resample_command(self, env_ids: Sequence[int]):
        # cube position in world frame (after the reset event has fired)
        cube_pos_w = self.object.data.root_pos_w[env_ids]
        # sample a lift height per env
        r = torch.empty(len(env_ids), device=self.device)
        lift_height = r.uniform_(*self.cfg.lift_height_range)
        # desired goal in world frame: directly above the cube
        desired_pos_w = cube_pos_w.clone()
        desired_pos_w[:, 2] = cube_pos_w[:, 2] + lift_height
        # convert to robot root frame for storage in pose_command_b
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        desired_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, desired_pos_w)
        self.pose_command_b[env_ids, :3] = desired_pos_b
        # orientation: sample from ranges (default zeros -> identity quat)
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class CubeAbovePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`CubeAbovePoseCommand`.

    Inherits ``ranges`` from the parent but only the roll/pitch/yaw fields
    are used (orientation sampling). The ``pos_x``/``pos_y``/``pos_z`` ranges
    on the parent are ignored — position is computed from the cube pose plus
    ``lift_height_range``.
    """

    class_type: type = CubeAbovePoseCommand

    object_name: str = "object"
    """Scene entity name of the cube/object to track."""

    lift_height_range: tuple[float, float] = (0.15, 0.20)
    """Range from which the per-env lift height (m, above cube) is sampled."""


__all__ = ["CubeAbovePoseCommand", "CubeAbovePoseCommandCfg"]
