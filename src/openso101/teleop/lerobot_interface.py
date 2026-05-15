# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Runtime LeRobot wrappers for Safe Sim2Real teleoperation."""

from __future__ import annotations

from typing import Any, Mapping

from .so101_mapping import lerobot_action_to_ordered_targets, transform_ordered_targets


class LeRobotDependencyError(RuntimeError):
    """Raised when LeRobot is needed but is not installed in the active env."""


def _make_so101_leader(port: str, robot_id: str) -> Any:
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig
    except ImportError as exc:
        raise LeRobotDependencyError(
            "LeRobot SO101 support is not installed. Install LeRobot in this "
            "environment before running Safe Sim2Real teleop."
        ) from exc

    return make_robot_from_config(SO101LeaderConfig(port=port, id=robot_id))


class LeRobotSO101Leader:
    """SO101 leader-arm reader with Safe Sim2Real target conversion."""

    def __init__(
        self,
        port: str,
        robot_id: str,
        robot: Any | None = None,
        *,
        inverted_joints=(),
        joint_offsets_rad: Mapping[str, float] | None = None,
    ):
        self.port = port
        self.robot_id = robot_id
        self._robot = robot
        self.inverted_joints = tuple(inverted_joints)
        self.joint_offsets_rad = dict(joint_offsets_rad or {})

    @property
    def robot(self) -> Any:
        if self._robot is None:
            self._robot = _make_so101_leader(self.port, self.robot_id)
        return self._robot

    def connect(self) -> None:
        self.robot.connect()

    def read_action(self) -> Mapping[str, float]:
        return self.robot.get_action()

    def read_ordered_targets(self) -> tuple[Mapping[str, float], tuple[float, ...]]:
        action = self.read_action()
        targets = lerobot_action_to_ordered_targets(action)
        targets = transform_ordered_targets(
            targets,
            inverted_joints=self.inverted_joints,
            offsets_rad=self.joint_offsets_rad,
        )
        return action, targets

    def read_target_tensor(self, device: str):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to create Isaac action tensors.") from exc

        action, targets = self.read_ordered_targets()
        return action, torch.tensor(targets, dtype=torch.float32, device=device)
