# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Lift task — pick up a cube from the table.

SKELETON: the real cfg body will be ported from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/`
(both `lift_env_cfg.py` and `joint_pos_env_cfg.py` collapsed into
this single class) once the legacy source is finalized.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from openso101.envs import OpenSO101EnvCfg


@configclass
class LiftEnvCfg(OpenSO101EnvCfg):
    """Concrete cfg for the Lift task.

    Variants (cameras / teleop / play) are applied by the framework's env
    factory via `configure_*` methods inherited from OpenSO101EnvCfg.
    Currently a SKELETON — `__post_init__` will populate scene,
    observations, actions, rewards, terminations during the real port.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        raise NotImplementedError(
            "LiftEnvCfg not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/lift/"
        )
