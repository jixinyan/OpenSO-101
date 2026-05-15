# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO101 leader-arm to simulated follower joint mapping.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py` lands. The
legacy implementation is under active revision (motor-unit remap,
streaming HDF5 recording); this skeleton freezes the public surface.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class JointLimitsDeg:
    """Closed joint-position interval in degrees."""

    lower: float
    upper: float


# Public constants — placeholder values matching declared types.
LEROBOT_SO101_ACTION_NAMES: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

SO101_TELEOP_CONTROL_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

LEROBOT_TO_SO101_CONTROL_JOINTS: dict[str, str] = dict(
    zip(LEROBOT_SO101_ACTION_NAMES, SO101_TELEOP_CONTROL_JOINT_NAMES, strict=True)
)

LEROBOT_LEADER_LIMITS_DEG: dict[str, JointLimitsDeg] = {
    "shoulder_pan.pos": JointLimitsDeg(-100.0, 100.0),
    "shoulder_lift.pos": JointLimitsDeg(-100.0, 100.0),
    "elbow_flex.pos": JointLimitsDeg(-100.0, 100.0),
    "wrist_flex.pos": JointLimitsDeg(-100.0, 100.0),
    "wrist_roll.pos": JointLimitsDeg(-100.0, 100.0),
    "gripper.pos": JointLimitsDeg(0.0, 100.0),
}

SO101_TELEOP_TARGET_LIMITS_DEG: dict[str, JointLimitsDeg] = {
    "shoulder_pan": JointLimitsDeg(-110.0, 110.0),
    "shoulder_lift": JointLimitsDeg(-100.0, 100.0),
    "elbow_flex": JointLimitsDeg(-100.0, 90.0),
    "wrist_flex": JointLimitsDeg(-95.0, 95.0),
    "wrist_roll": JointLimitsDeg(-160.0, 160.0),
    "gripper": JointLimitsDeg(-10.0, 100.0),
}

# Sim radians <-> motor-units conversion constants.
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "Rotation":    (-1.91986, 1.91986),
    "Pitch":       (-1.74533, 1.74533),
    "Elbow":       (-1.69, 1.69),
    "Wrist_Pitch": (-1.65806, 1.65806),
    "Wrist_Roll":  (-2.74385, 2.84121),
    "Jaw":         (-0.174533, 1.74533),
}

MOTOR_MIN: float = -100.0
MOTOR_MAX: float = 100.0

JOINT_ORDER: tuple[str, ...] = (
    "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"
)


def get_teleop_target_limits() -> dict[str, JointLimitsDeg]:
    """Return the canonical target limits for the local SO101 USD follower."""
    raise NotImplementedError(
        "get_teleop_target_limits not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def get_sim_joint_names() -> tuple[str, ...]:
    """Return the canonical SO101 simulator joint names."""
    raise NotImplementedError(
        "get_sim_joint_names not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def lerobot_action_to_joint_targets(action: Mapping[str, float]) -> dict[str, float]:
    """Convert a LeRobot SO101 action dict to Safe Sim2Real radian targets."""
    raise NotImplementedError(
        "lerobot_action_to_joint_targets not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def lerobot_action_to_ordered_targets(action: Mapping[str, float]) -> tuple[float, ...]:
    """Return targets ordered for the Safe Sim2Real SO-ARM101 action vector."""
    raise NotImplementedError(
        "lerobot_action_to_ordered_targets not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def parse_joint_name_set(value: str | None) -> frozenset[str]:
    """Parse a comma-separated SO-101 joint-name list."""
    raise NotImplementedError(
        "parse_joint_name_set not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def parse_joint_offsets_deg(value: str | None) -> dict[str, float]:
    """Parse comma-separated ``joint:offset_deg`` calibration offsets."""
    raise NotImplementedError(
        "parse_joint_offsets_deg not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def transform_ordered_targets(
    targets: Iterable[float],
    *,
    inverted_joints: Iterable[str] = (),
    offsets_rad: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """Apply optional per-joint sign flips and radian offsets to ordered targets."""
    raise NotImplementedError(
        "transform_ordered_targets not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def sim_radians_to_motor_units(rad: "torch.Tensor", joint_name: str) -> "torch.Tensor":
    """Map per-joint radians -> motor units in [-100, 100]."""
    raise NotImplementedError(
        "sim_radians_to_motor_units not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def motor_units_to_sim_radians(motor: "torch.Tensor", joint_name: str) -> "torch.Tensor":
    """Inverse of sim_radians_to_motor_units."""
    raise NotImplementedError(
        "motor_units_to_sim_radians not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def batched_action_to_motor_units(actions: "torch.Tensor") -> "torch.Tensor":
    """Remap ``(..., 6)`` sim-radian actions/states -> motor units."""
    raise NotImplementedError(
        "batched_action_to_motor_units not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


def batched_motor_units_to_action(motors: "torch.Tensor") -> "torch.Tensor":
    """Inverse of batched_action_to_motor_units."""
    raise NotImplementedError(
        "batched_motor_units_to_action not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/so101_mapping.py"
    )


__all__ = [
    "JointLimitsDeg",
    "LEROBOT_SO101_ACTION_NAMES",
    "SO101_TELEOP_CONTROL_JOINT_NAMES",
    "LEROBOT_TO_SO101_CONTROL_JOINTS",
    "LEROBOT_LEADER_LIMITS_DEG",
    "SO101_TELEOP_TARGET_LIMITS_DEG",
    "JOINT_LIMITS_RAD",
    "MOTOR_MIN",
    "MOTOR_MAX",
    "JOINT_ORDER",
    "get_teleop_target_limits",
    "get_sim_joint_names",
    "lerobot_action_to_joint_targets",
    "lerobot_action_to_ordered_targets",
    "parse_joint_name_set",
    "parse_joint_offsets_deg",
    "transform_ordered_targets",
    "sim_radians_to_motor_units",
    "motor_units_to_sim_radians",
    "batched_action_to_motor_units",
    "batched_motor_units_to_action",
]
