# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PickPlace task-local MDP terminations.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/terminations.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def curriculum_complete(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    threshold: float = 0.03,
    robot_cfg: "SceneEntityCfg" = None,
    object_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Episode succeeds when final-stage cube surface touches the goal sphere."""
    raise NotImplementedError(
        "curriculum_complete not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/terminations.py"
    )


__all__ = [
    "curriculum_complete",
]
