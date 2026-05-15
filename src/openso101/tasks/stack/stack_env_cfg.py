# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Stack task — stack one cube on top of another.

SKELETON: the real cfg will be ported from
`/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/`
(both `stack_env_cfg.py` and `joint_pos_env_cfg.py` collapsed into
this single class) once the legacy source is finalized.

The real port wires in physics DR via
`openso101.sim2real.domain_randomization.attach_all_physics_dr`.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from openso101.envs import OpenSO101EnvCfg


@configclass
class StackEnvCfg(OpenSO101EnvCfg):
    """Concrete cfg for the Stack task.

    SKELETON — `__post_init__` populates scene/MDP during the real port.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        raise NotImplementedError(
            "StackEnvCfg not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/tasks/composite/stack/"
        )
