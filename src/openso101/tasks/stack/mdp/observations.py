# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task-local MDP observation functions.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/observations.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Position of an object expressed in the robot's root frame."""
    raise NotImplementedError(
        "object_position_in_robot_root_frame not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/observations.py"
    )


def cube_top_to_cube_bottom_offset(
    env: "ManagerBasedRLEnv",
    cube_height: float = 0.03,
    cube_top_cfg: "SceneEntityCfg" = None,
    cube_bottom_cfg: "SceneEntityCfg" = None,
    robot_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Vector from cube_top to the stacking goal (top of cube_bottom) in robot root frame."""
    raise NotImplementedError(
        "cube_top_to_cube_bottom_offset not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/observations.py"
    )


__all__ = [
    "object_position_in_robot_root_frame",
    "cube_top_to_cube_bottom_offset",
]
