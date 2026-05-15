# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task-local MDP reward functions.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def get_cube_top_was_lifted(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return the per-env episode flag for a controlled cube-top lift."""
    raise NotImplementedError(
        "get_cube_top_was_lifted not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py"
    )


def ee_to_cube_top_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    cube_top_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward for moving the end-effector close to cube_top (tanh kernel)."""
    raise NotImplementedError(
        "ee_to_cube_top_distance not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py"
    )


def cube_top_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    grasp_distance_threshold: float = 0.07,
    closed_threshold: float = 0.12,
    robot_cfg: "SceneEntityCfg" = None,
    cube_top_cfg: "SceneEntityCfg" = None,
    ee_frame_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Binary reward when cube_top is lifted while controlled by the gripper."""
    raise NotImplementedError(
        "cube_top_is_lifted not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py"
    )


def cubes_xy_aligned(
    env: "ManagerBasedRLEnv",
    std: float,
    cube_top_cfg: "SceneEntityCfg" = None,
    cube_bottom_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward for aligning cube_top over cube_bottom in the xy-plane (tanh kernel)."""
    raise NotImplementedError(
        "cubes_xy_aligned not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py"
    )


def cubes_stacked(
    env: "ManagerBasedRLEnv",
    std: float,
    cube_height: float = 0.03,
    cube_top_cfg: "SceneEntityCfg" = None,
    cube_bottom_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Reward when cube_top is resting stably on top of cube_bottom."""
    raise NotImplementedError(
        "cubes_stacked not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py"
    )


def cube_released_on_stack(
    env: "ManagerBasedRLEnv",
    xy_threshold: float,
    height_tolerance: float,
    cube_height: float = 0.03,
    open_threshold: float = 0.35,
    object_static_threshold: float = 0.2,
    robot_cfg: "SceneEntityCfg" = None,
    cube_top_cfg: "SceneEntityCfg" = None,
    cube_bottom_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Terminal-style reward for releasing a settled cube_top on cube_bottom."""
    raise NotImplementedError(
        "cube_released_on_stack not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/rewards.py"
    )


__all__ = [
    "get_cube_top_was_lifted",
    "ee_to_cube_top_distance",
    "cube_top_is_lifted",
    "cubes_xy_aligned",
    "cubes_stacked",
    "cube_released_on_stack",
]
