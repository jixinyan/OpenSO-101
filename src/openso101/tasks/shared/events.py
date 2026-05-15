# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Custom reset events for atomic skills that start from non-default states.

For skills like Lift, Transport, and Place the robot must begin with the object
already grasped.  The helpers below set joint positions to a known configuration
and teleport the object to the approximate end-effector location so that the
closed gripper holds it from the first simulation step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_to_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_targets: dict[str, float],
    position_noise: float = 0.0,
    velocity_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot joints to absolute target positions with optional noise.

    Args:
        position_targets: Mapping from joint name to target position (rad).
        position_noise: Uniform noise half-range added to each target.
        velocity_noise: Uniform noise half-range for joint velocities (centred on 0).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    for joint_name, target in position_targets.items():
        joint_ids, _ = robot.find_joints(joint_name)
        noise = (torch.rand(len(env_ids), len(joint_ids), device=env.device) * 2 - 1) * position_noise
        joint_pos[:, joint_ids] = target + noise

    if velocity_noise > 0.0:
        joint_vel = (torch.rand_like(joint_vel) * 2 - 1) * velocity_noise

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_object_to_fixed_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position: tuple[float, float, float],
    position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Teleport the object to a fixed world position with optional noise.

    Useful for placing the object where the gripper will be after
    ``reset_joints_to_positions`` so that the object starts grasped.
    """
    obj: RigidObject = env.scene[asset_cfg.name]

    # Build default root state (pos, quat, lin_vel, ang_vel) = 13 dims
    root_state = obj.data.default_root_state[env_ids].clone()

    pos = torch.tensor(position, device=env.device, dtype=torch.float32)
    noise_range = torch.tensor(position_noise, device=env.device, dtype=torch.float32)
    noise = (torch.rand(len(env_ids), 3, device=env.device) * 2 - 1) * noise_range

    # Add env origin offsets (multi-env)
    root_state[:, :3] = pos + noise + env.scene.env_origins[env_ids, :3]
    # Zero velocities
    root_state[:, 7:] = 0.0

    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_robot_with_random_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    offset_range: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot to default pose + uniform random joint offsets.

    Used by the ReturnHome skill so the policy must learn to reach default
    from an arbitrary (but reachable) configuration.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    offsets = (torch.rand_like(joint_pos) * 2 - 1) * offset_range
    # Don't randomize the gripper (last joint) — keep it open
    offsets[:, -1] = 0.0
    joint_pos = joint_pos + offsets

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_grasped_object_and_arm(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    arm_joint_targets: dict[str, float],
    ee_preset_base_pos: tuple[float, float, float],
    pan_range: tuple[float, float] = (-0.4, 0.4),
    reach_noise: float = 0.05,
    object_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset so that the arm is holding the object at a varied location.

    Samples ``shoulder_pan`` uniformly from ``pan_range`` and rotates both the
    arm preset and the cube world position by the same angle about the base Z
    axis.  All other arm joints are set from ``arm_joint_targets`` (plus small
    ``reach_noise`` on shoulder_lift/elbow_flex/wrist_flex) so the gripper
    always ends up approximately at the rotated ``ee_preset_base_pos`` — with
    the cube co-located inside the closed gripper.

    Args:
        arm_joint_targets: Absolute targets for arm joints *at pan=0*, keyed by
            joint name.  Must include ``gripper`` (closed, typically 0.0).
            ``shoulder_pan`` is ignored if present (it is sampled).
        ee_preset_base_pos: End-effector world position when ``pan=0`` and the
            rest of the arm is at ``arm_joint_targets``.  This is the
            reference point the cube is teleported to (then rotated by pan).
        pan_range: Uniform sample range for ``shoulder_pan`` (radians).
        reach_noise: Uniform half-range (radians) added to shoulder_lift,
            elbow_flex, wrist_flex for diversity in arm reach / height.
        object_pos_offset: Extra offset added to cube world pos, e.g. for
            Place-style starts where the cube is elevated above the gripper.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    n = len(env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    pan_low, pan_high = pan_range
    pan = torch.rand(n, device=env.device) * (pan_high - pan_low) + pan_low

    _reach_jitter_joints = {"Pitch", "Elbow", "Wrist_Pitch"}
    for jname, target in arm_joint_targets.items():
        if jname == "Rotation":
            continue
        jids, _ = robot.find_joints(jname)
        if reach_noise > 0.0 and jname in _reach_jitter_joints:
            noise = (torch.rand(n, len(jids), device=env.device) * 2 - 1) * reach_noise
        else:
            noise = 0.0
        joint_pos[:, jids] = target + noise

    pan_ids, _ = robot.find_joints("Rotation")
    joint_pos[:, pan_ids] = pan.unsqueeze(-1)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    ee_x, ee_y, ee_z = ee_preset_base_pos
    cos_p = torch.cos(pan)
    sin_p = torch.sin(pan)
    cube_x = ee_x * cos_p - ee_y * sin_p + object_pos_offset[0]
    cube_y = ee_x * sin_p + ee_y * cos_p + object_pos_offset[1]
    cube_z = torch.full_like(pan, ee_z + object_pos_offset[2])

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, 0] = cube_x + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = cube_y + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = cube_z + env.scene.env_origins[env_ids, 2]
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


__all__ = [
    "reset_joints_to_positions",
    "reset_object_to_fixed_position",
    "reset_robot_with_random_joints",
    "reset_grasped_object_and_arm",
]
