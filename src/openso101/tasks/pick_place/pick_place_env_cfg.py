# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PickPlace task — pick up a cube and place it at a goal pose.

SKELETON: the real cfg (with 3-stage curriculum) will be ported from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/`.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from openso101.envs import OpenSO101EnvCfg


@configclass
class PickPlaceEnvCfg(OpenSO101EnvCfg):
    """Concrete cfg for the PickPlace task.

    SKELETON — `__post_init__` populates scene/MDP during the real port.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        raise NotImplementedError(
            "PickPlaceEnvCfg not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/pick_and_place/"
        )
