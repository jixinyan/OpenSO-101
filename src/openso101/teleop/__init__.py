# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Teleoperation adapters for OpenSO-101."""

from openso101.robots import SO101_SIM_JOINT_NAMES

from .so101_mapping import (
    LEROBOT_SO101_ACTION_NAMES,
    SO101_TELEOP_CONTROL_JOINT_NAMES,
    lerobot_action_to_joint_targets,
    lerobot_action_to_ordered_targets,
)

__all__ = [
    "LEROBOT_SO101_ACTION_NAMES",
    "SO101_SIM_JOINT_NAMES",
    "SO101_TELEOP_CONTROL_JOINT_NAMES",
    "lerobot_action_to_joint_targets",
    "lerobot_action_to_ordered_targets",
]
