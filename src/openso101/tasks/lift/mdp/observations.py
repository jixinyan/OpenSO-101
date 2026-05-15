# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task-local MDP observation functions.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/observations.py` lands.
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
    """The position of the object in the robot's root frame."""
    raise NotImplementedError(
        "object_position_in_robot_root_frame not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/mdp/observations.py"
    )


__all__ = ["object_position_in_robot_root_frame"]
