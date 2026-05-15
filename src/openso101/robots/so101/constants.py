# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Pure SO-101 robot constants shared by RL tasks and teleoperation.

SKELETON: real values will be ported from
`/data/safe_sim2real/src/safe_sim2real/robots/so101_constants.py` once
the source is finalized. Until then the names are declared with
placeholder values so dependent code can import them.
"""

SO101_ARM_JOINT_NAMES: tuple[str, ...] = ()
SO101_GRIPPER_JOINT_NAME: str = ""
SO101_GRIPPER_JOINT_NAMES: tuple[str, ...] = ()
SO101_SIM_JOINT_NAMES: tuple[str, ...] = ()
SO101_GRIPPER_OPEN_POS: float = 0.0
SO101_GRIPPER_CLOSED_POS: float = 0.0
SO101_DEFAULT_JOINT_POS: dict[str, float] = {}
