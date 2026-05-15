# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Runtime LeRobot wrappers for SO-101 teleoperation.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py` lands. The
legacy implementation is under active revision (motor-unit remap,
streaming HDF5 recording); this skeleton freezes the public surface.
"""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class LeRobotDependencyError(RuntimeError):
    """Raised when LeRobot is needed but is not installed in the active env."""


class LeRobotSO101Leader:
    """SO101 leader-arm reader with SO-101 target conversion.

    SKELETON — instantiation raises NotImplementedError.
    """

    def __init__(
        self,
        port: str,
        robot_id: str,
        robot: Any | None = None,
        *,
        inverted_joints=(),
        joint_offsets_rad: Mapping[str, float] | None = None,
    ):
        raise NotImplementedError(
            "LeRobotSO101Leader not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py"
        )

    @property
    def robot(self) -> Any:
        """Lazily-created LeRobot SO101 leader robot instance."""
        raise NotImplementedError(
            "LeRobotSO101Leader.robot not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py"
        )

    def connect(self) -> None:
        """Connect to the physical SO101 leader arm."""
        raise NotImplementedError(
            "LeRobotSO101Leader.connect not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py"
        )

    def read_action(self) -> Mapping[str, float]:
        """Read the current raw LeRobot action dict from the leader arm."""
        raise NotImplementedError(
            "LeRobotSO101Leader.read_action not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py"
        )

    def read_ordered_targets(self) -> tuple[Mapping[str, float], tuple[float, ...]]:
        """Read and convert the leader action to ordered radian targets."""
        raise NotImplementedError(
            "LeRobotSO101Leader.read_ordered_targets not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py"
        )

    def read_target_tensor(self, device: str) -> tuple[Mapping[str, float], Any]:
        """Read the leader action as an Isaac-ready torch tensor on ``device``."""
        raise NotImplementedError(
            "LeRobotSO101Leader.read_target_tensor not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_interface.py"
        )


__all__ = [
    "LeRobotDependencyError",
    "LeRobotSO101Leader",
]
