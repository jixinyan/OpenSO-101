# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PickPlace task-local MDP grasp predicates.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/grasp.py` lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv


def object_grasped_by_jaws(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> "torch.Tensor":
    """Return a per-env bool tensor: True iff both jaws are in contact with the cube."""
    raise NotImplementedError(
        "object_grasped_by_jaws not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/grasp.py"
    )


def grasped_reward(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> "torch.Tensor":
    """Float version of :func:`object_grasped_by_jaws`, suitable as a RewTerm function."""
    raise NotImplementedError(
        "grasped_reward not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/mdp/grasp.py"
    )


__all__ = [
    "object_grasped_by_jaws",
    "grasped_reward",
]
