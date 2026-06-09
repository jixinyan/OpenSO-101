# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO101 leader-arm to simulated follower joint mapping.

LeRobot's SO101 leader reports joint positions in degrees using feature names
such as ``shoulder_pan.pos``. OpenSO-101 commands the Isaac articulation in
radians using the canonical SO101 USD joint names. The conversion here is
intentionally pure Python so the sim/real boundary can be tested without Isaac
Sim or hardware.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Mapping

from openso101.robots import SO101_SIM_JOINT_NAMES

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class JointLimitsDeg:
    """Closed joint-position interval in degrees."""

    lower: float
    upper: float


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

# LeRobot SO101 leader convention used by the upstream teleop project: arm
# joints report roughly [-100, 100] degrees and gripper reports [0, 100].
LEROBOT_LEADER_LIMITS_DEG: dict[str, JointLimitsDeg] = {
    "shoulder_pan.pos": JointLimitsDeg(-100.0, 100.0),
    "shoulder_lift.pos": JointLimitsDeg(-100.0, 100.0),
    "elbow_flex.pos": JointLimitsDeg(-100.0, 100.0),
    "wrist_flex.pos": JointLimitsDeg(-100.0, 100.0),
    "wrist_roll.pos": JointLimitsDeg(-100.0, 100.0),
    "gripper.pos": JointLimitsDeg(0.0, 100.0),
}

# Canonical target limits for the local SO101 USD follower.
SO101_TELEOP_TARGET_LIMITS_DEG: dict[str, JointLimitsDeg] = {
    "shoulder_pan": JointLimitsDeg(-110.0, 110.0),
    "shoulder_lift": JointLimitsDeg(-100.0, 100.0),
    "elbow_flex": JointLimitsDeg(-100.0, 90.0),
    "wrist_flex": JointLimitsDeg(-95.0, 95.0),
    "wrist_roll": JointLimitsDeg(-160.0, 160.0),
    "gripper": JointLimitsDeg(-10.0, 100.0),
}


def get_teleop_target_limits() -> dict[str, JointLimitsDeg]:
    return SO101_TELEOP_TARGET_LIMITS_DEG


def get_sim_joint_names() -> tuple[str, ...]:
    return SO101_SIM_JOINT_NAMES


def _clamp(value: float, limits: JointLimitsDeg) -> float:
    return min(max(value, limits.lower), limits.upper)


def _map_interval(value: float, source: JointLimitsDeg, target: JointLimitsDeg) -> float:
    if source.upper == source.lower:
        raise ValueError("source limits must span a non-zero interval")
    normalized = (_clamp(value, source) - source.lower) / (source.upper - source.lower)
    return target.lower + normalized * (target.upper - target.lower)


def lerobot_action_to_joint_targets(action: Mapping[str, float]) -> dict[str, float]:
    """Convert a LeRobot SO101 action dict to OpenSO-101 radian targets."""

    missing = [name for name in LEROBOT_SO101_ACTION_NAMES if name not in action]
    if missing:
        missing_names = ", ".join(missing)
        raise KeyError(f"LeRobot action is missing required SO101 joint(s): {missing_names}")

    targets: dict[str, float] = {}
    for action_name in LEROBOT_SO101_ACTION_NAMES:
        joint_name = LEROBOT_TO_SO101_CONTROL_JOINTS[action_name]
        target_deg = _map_interval(
            float(action[action_name]),
            LEROBOT_LEADER_LIMITS_DEG[action_name],
            SO101_TELEOP_TARGET_LIMITS_DEG[joint_name],
        )
        targets[joint_name] = math.radians(target_deg)
    return targets


def lerobot_action_to_ordered_targets(action: Mapping[str, float]) -> tuple[float, ...]:
    """Return targets ordered for the OpenSO-101 SO-ARM101 action vector."""

    targets = lerobot_action_to_joint_targets(action)
    return tuple(targets[joint_name] for joint_name in SO101_TELEOP_CONTROL_JOINT_NAMES)


def parse_joint_name_set(value: str | None) -> frozenset[str]:
    """Parse a comma-separated SO-101 joint-name list."""

    if value is None or not value.strip():
        return frozenset()
    names = frozenset(name.strip() for name in value.split(",") if name.strip())
    unknown = sorted(names.difference(SO101_TELEOP_CONTROL_JOINT_NAMES))
    if unknown:
        raise ValueError(f"Unknown SO-101 joint name(s): {', '.join(unknown)}")
    return names


def parse_joint_offsets_deg(value: str | None) -> dict[str, float]:
    """Parse comma-separated ``joint:offset_deg`` calibration offsets."""

    if value is None or not value.strip():
        return {}
    offsets: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected joint:offset_deg entry, got {item!r}")
        name, raw_offset = (part.strip() for part in item.split(":", 1))
        if name not in SO101_TELEOP_CONTROL_JOINT_NAMES:
            raise ValueError(f"Unknown SO-101 joint name: {name}")
        offsets[name] = math.radians(float(raw_offset))
    return offsets


def transform_ordered_targets(
    targets: Iterable[float],
    *,
    inverted_joints: Iterable[str] = (),
    offsets_rad: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """Apply optional per-joint sign flips and radian offsets to ordered targets."""

    inverted = set(inverted_joints)
    offsets = offsets_rad or {}
    transformed: list[float] = []
    for joint_name, target in zip(SO101_TELEOP_CONTROL_JOINT_NAMES, targets, strict=True):
        value = -float(target) if joint_name in inverted else float(target)
        value += float(offsets.get(joint_name, 0.0))
        limits = SO101_TELEOP_TARGET_LIMITS_DEG[joint_name]
        lower = math.radians(limits.lower)
        upper = math.radians(limits.upper)
        transformed.append(min(max(value, lower), upper))
    return tuple(transformed)


# ---------------------------------------------------------------------------
# Sim radians <-> LeRobot motor-units conversion (sim-to-real IL transfer).
#
# LeRobot's STS3215 motor driver uses normalized [-100, 100] motor units.
# Sim records joint targets/states in radians. For an IL policy trained on
# sim data to deploy on real SO101 hardware, the recorded action and
# observation.state must be in the SAME UNITS the real driver expects --
# i.e., motor units, not radians. The functions below are the single source
# of truth for that conversion.
#
# Per-joint limit table matches the LeRobot leader-arm calibration JSON
# convention (derived from SO101 URDF joint limits). Adjust ``JOINT_LIMITS_RAD``
# if your physical SO101 has been calibrated to a non-default range; the
# inverse mapping preserves the same limits.
# ---------------------------------------------------------------------------

# (rad_min, rad_max) per joint.
#
# Values are exact from ``robots/trs_so101/urdf/so_arm101.urdf`` (URDF is the
# single source of truth for joint ranges). The URDF -> sim joint-name mapping
# is: shoulder_pan -> Rotation, shoulder_lift -> Pitch, elbow_flex -> Elbow,
# wrist_flex -> Wrist_Pitch, wrist_roll -> Wrist_Roll, gripper -> Jaw.
#
# Asymmetric ranges (Wrist_Roll, Jaw) are intentional and match the URDF
# source; do not symmetrize them. The Wrist_Roll upper bound (2.84121) lies
# outside the previously-used symmetric (-2.74, 2.74) interval -- sim values
# in [2.74, 2.84121] would otherwise extrapolate past MOTOR_MAX=100.
#
# To recalibrate for a specific physical robot, override these values via a
# calibration JSON file (future work, not in this commit).
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "Rotation":    (-1.91986, 1.91986),
    "Pitch":       (-1.74533, 1.74533),
    "Elbow":       (-1.69, 1.69),
    "Wrist_Pitch": (-1.65806, 1.65806),
    "Wrist_Roll":  (-2.74385, 2.84121),   # asymmetric (per URDF)
    "Jaw":         (-0.174533, 1.74533),  # asymmetric: over-close to full-open
}

# Motor-unit endpoint convention (matches LeRobot STS3215).
MOTOR_MIN = -100.0
MOTOR_MAX = 100.0

# Joint ordering for batched-tensor APIs. Matches the sim action layout
# in safe_sim2real (5 arm joints, then gripper).
JOINT_ORDER: tuple[str, ...] = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")


def sim_radians_to_motor_units(rad: "torch.Tensor", joint_name: str) -> "torch.Tensor":
    """Map per-joint radians -> motor units in [-100, 100].

    Inputs outside ``[rad_lo, rad_hi]`` are clamped to the joint's valid range
    so the recorded motor-unit value never exceeds ``[MOTOR_MIN, MOTOR_MAX]``.
    This mirrors the clamping behavior in the legacy leader-mapping API.
    """
    import torch  # local import to keep module importable without torch at top level

    rad_lo, rad_hi = JOINT_LIMITS_RAD[joint_name]
    rad = torch.clamp(rad, min=rad_lo, max=rad_hi)
    return MOTOR_MIN + (rad - rad_lo) * (MOTOR_MAX - MOTOR_MIN) / (rad_hi - rad_lo)


def motor_units_to_sim_radians(motor: "torch.Tensor", joint_name: str) -> "torch.Tensor":
    """Inverse of :func:`sim_radians_to_motor_units`.

    Inputs outside ``[MOTOR_MIN, MOTOR_MAX]`` are clamped so the recovered
    radian value never exceeds the joint's URDF range.
    """
    import torch  # local import to keep module importable without torch at top level

    rad_lo, rad_hi = JOINT_LIMITS_RAD[joint_name]
    motor = torch.clamp(motor, min=MOTOR_MIN, max=MOTOR_MAX)
    return rad_lo + (motor - MOTOR_MIN) * (rad_hi - rad_lo) / (MOTOR_MAX - MOTOR_MIN)


def batched_action_to_motor_units(actions: "torch.Tensor") -> "torch.Tensor":
    """Remap ``(..., 6)`` sim-radian actions/states -> motor units.

    Joint order matches :data:`JOINT_ORDER`; the gripper is the last column.
    """
    import torch  # local import to keep module importable without torch at top level

    if actions.shape[-1] != len(JOINT_ORDER):
        raise ValueError(
            f"Expected last dim={len(JOINT_ORDER)} (got {tuple(actions.shape)}); "
            f"joints: {JOINT_ORDER}"
        )
    out = torch.empty_like(actions)
    for i, name in enumerate(JOINT_ORDER):
        out[..., i] = sim_radians_to_motor_units(actions[..., i], name)
    return out


def batched_motor_units_to_action(motors: "torch.Tensor") -> "torch.Tensor":
    """Inverse of :func:`batched_action_to_motor_units`."""
    import torch  # local import to keep module importable without torch at top level

    if motors.shape[-1] != len(JOINT_ORDER):
        raise ValueError(
            f"Expected last dim={len(JOINT_ORDER)} (got {tuple(motors.shape)}); "
            f"joints: {JOINT_ORDER}"
        )
    out = torch.empty_like(motors)
    for i, name in enumerate(JOINT_ORDER):
        out[..., i] = motor_units_to_sim_radians(motors[..., i], name)
    return out
