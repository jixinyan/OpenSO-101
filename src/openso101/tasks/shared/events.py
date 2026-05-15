# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Shared MDP event functions used by built-in manipulation tasks.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/shared/events.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def reset_joints_to_positions(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
    position_targets: dict[str, float],
    position_noise: float = 0.0,
    velocity_noise: float = 0.0,
    asset_cfg: "SceneEntityCfg" = None,
) -> None:
    """Reset robot joints to absolute target positions with optional noise."""
    raise NotImplementedError(
        "reset_joints_to_positions not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/events.py"
    )


def reset_object_to_fixed_position(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
    position: tuple[float, float, float],
    position_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: "SceneEntityCfg" = None,
) -> None:
    """Teleport the object to a fixed world position with optional noise."""
    raise NotImplementedError(
        "reset_object_to_fixed_position not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/events.py"
    )


def reset_robot_with_random_joints(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
    offset_range: float = 0.5,
    asset_cfg: "SceneEntityCfg" = None,
) -> None:
    """Reset robot to default pose + uniform random joint offsets."""
    raise NotImplementedError(
        "reset_robot_with_random_joints not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/events.py"
    )


def reset_grasped_object_and_arm(
    env: "ManagerBasedEnv",
    env_ids: "torch.Tensor",
    arm_joint_targets: dict[str, float],
    ee_preset_base_pos: tuple[float, float, float],
    pan_range: tuple[float, float] = (-0.4, 0.4),
    reach_noise: float = 0.05,
    object_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> None:
    """Reset so that the arm is holding the object at a varied location."""
    raise NotImplementedError(
        "reset_grasped_object_and_arm not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/shared/events.py"
    )


__all__ = [
    "reset_joints_to_positions",
    "reset_object_to_fixed_position",
    "reset_robot_with_random_joints",
    "reset_grasped_object_and_arm",
]
