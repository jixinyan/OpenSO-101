# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO-101 robot module (constants always eager; ArticulationCfg + cameras lazy)."""

from .so101.constants import (
    SO101_ARM_JOINT_NAMES,
    SO101_DEFAULT_JOINT_POS,
    SO101_GRIPPER_CLOSED_POS,
    SO101_GRIPPER_JOINT_NAME,
    SO101_GRIPPER_JOINT_NAMES,
    SO101_GRIPPER_OPEN_POS,
    SO101_SIM_JOINT_NAMES,
)

__all__ = [
    "SO101_ARM_JOINT_NAMES",
    "SO101_DEFAULT_JOINT_POS",
    "SO101_GRIPPER_CLOSED_POS",
    "SO101_GRIPPER_JOINT_NAME",
    "SO101_GRIPPER_JOINT_NAMES",
    "SO101_GRIPPER_OPEN_POS",
    "SO101_SIM_JOINT_NAMES",
    "SO_ARM101_CFG",
    "SO_ARM101_TELEOP_CFG",
    "so101_usd_path",
    "overhead_camera_cfg",
    "wrist_camera_cfg",
]


def __getattr__(name: str):
    if name in {
        "SO_ARM101_CFG",
        "SO_ARM101_TELEOP_CFG",
        "so101_usd_path",
    }:
        from .so101.so_arm101 import (
            SO_ARM101_CFG,
            SO_ARM101_TELEOP_CFG,
            so101_usd_path,
        )

        values = {
            "SO_ARM101_CFG": SO_ARM101_CFG,
            "SO_ARM101_TELEOP_CFG": SO_ARM101_TELEOP_CFG,
            "so101_usd_path": so101_usd_path,
        }
        return values[name]
    if name in {"overhead_camera_cfg", "wrist_camera_cfg"}:
        from .so101.cameras import overhead_camera_cfg, wrist_camera_cfg

        values = {
            "overhead_camera_cfg": overhead_camera_cfg,
            "wrist_camera_cfg": wrist_camera_cfg,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
