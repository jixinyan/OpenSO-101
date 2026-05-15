# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Pure SO101 robot constants shared by RL tasks and teleoperation."""

SO101_ARM_JOINT_NAMES: tuple[str, ...] = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
)
SO101_GRIPPER_JOINT_NAME = "Jaw"
SO101_GRIPPER_JOINT_NAMES: tuple[str, ...] = (SO101_GRIPPER_JOINT_NAME,)
SO101_SIM_JOINT_NAMES: tuple[str, ...] = (*SO101_ARM_JOINT_NAMES, SO101_GRIPPER_JOINT_NAME)

SO101_GRIPPER_OPEN_POS = 0.8
SO101_GRIPPER_CLOSED_POS = 0.0
"""Operational gripper command range for cube grasping.

Physical joint limits are roughly ``[-0.174, 1.745]`` (full over-close to
full open), but for a 5cm cube the parallel-jaw operational range is
``[0.0, 0.8]`` -- 0.8 rad opens wide enough to clear the cube, 0.0 rad
pinches the jaws parallel. The smaller per-step jaw swing keeps the
``BinaryJointPositionAction`` open<->close transition under the gripper
actuator's ``effort_limit_sim`` cap."""
SO101_DEFAULT_JOINT_POS: dict[str, float] = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745,
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    SO101_GRIPPER_JOINT_NAME: -0.1465,
}
