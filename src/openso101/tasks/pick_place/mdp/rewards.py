# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PickPlace task-local MDP rewards.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/rewards.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def cube_to_curriculum_stage_goal(
    env: "ManagerBasedRLEnv",
    stage: int,
    std: float,
    minimal_height: "float | None" = 0.04,
    command_name: str = "object_pose",
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Tanh-kernel reward to one stage goal, active only during that stage."""
    raise NotImplementedError(
        "cube_to_curriculum_stage_goal not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/rewards.py"
    )


def stage_completion_bonus(
    env: "ManagerBasedRLEnv",
    completed_stage: int,
    command_name: str = "object_pose",
) -> "torch.Tensor":
    """One-step sparse reward when touching a stage goal advances the command."""
    raise NotImplementedError(
        "stage_completion_bonus not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/rewards.py"
    )


def object_is_lifted_when_grasped(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    force_threshold: float = 0.5,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """``object_is_lifted`` gated on contact-confirmed grasp."""
    raise NotImplementedError(
        "object_is_lifted_when_grasped not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/rewards.py"
    )


def cube_to_curriculum_stage_goal_when_grasped(
    env: "ManagerBasedRLEnv",
    stage: int,
    std: float,
    minimal_height: "float | None" = 0.04,
    force_threshold: float = 0.5,
    command_name: str = "object_pose",
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """:func:`cube_to_curriculum_stage_goal` multiplicatively gated on grasp."""
    raise NotImplementedError(
        "cube_to_curriculum_stage_goal_when_grasped not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/rewards.py"
    )


def cube_to_curriculum_stage_goal_height_gated(
    env: "ManagerBasedRLEnv",
    stage: int,
    std: float,
    minimal_height: float,
    command_name: str = "object_pose",
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Stage-specific goal-distance reward, gated only on object height."""
    raise NotImplementedError(
        "cube_to_curriculum_stage_goal_height_gated not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/rewards.py"
    )


__all__ = [
    "cube_to_curriculum_stage_goal",
    "stage_completion_bonus",
    "object_is_lifted_when_grasped",
    "cube_to_curriculum_stage_goal_when_grasped",
    "cube_to_curriculum_stage_goal_height_gated",
]
