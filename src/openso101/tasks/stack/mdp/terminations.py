# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task-local MDP termination functions.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/terminations.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def cube_dropped(
    env: "ManagerBasedRLEnv",
    minimum_height: float = -0.05,
    cube_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Terminate when a cube falls below the table surface."""
    raise NotImplementedError(
        "cube_dropped not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/terminations.py"
    )


def cubes_stacked_success(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.015,
    height_tolerance: float = 0.01,
    cube_height: float = 0.03,
    open_threshold: float = 0.35,
    object_static_threshold: float = 0.2,
    robot_cfg: "SceneEntityCfg" = None,
    cube_top_cfg: "SceneEntityCfg" = None,
    cube_bottom_cfg: "SceneEntityCfg" = None,
    gripper_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Terminate when cube_top was lifted, released, stacked, and both cubes settle."""
    raise NotImplementedError(
        "cubes_stacked_success not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/mdp/terminations.py"
    )


__all__ = [
    "cube_dropped",
    "cubes_stacked_success",
]
