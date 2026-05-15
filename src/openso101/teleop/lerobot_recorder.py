# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""LeRobot dataset recording helpers for SO-101 teleoperation.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py` lands. The
legacy implementation is under active revision (motor-unit remap,
streaming HDF5 recording); this skeleton freezes the public surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


REQUIRED_CAMERA_NAMES: tuple[str, str] = ("wrist_camera", "overhead_camera")
REQUIRED_REOPEN_METADATA: tuple[Path, ...] = (
    Path("meta/info.json"),
    Path("meta/tasks.parquet"),
)


def has_lerobot_metadata(root: str | Path) -> bool:
    """Return whether ``root`` has enough local metadata for LeRobot to reopen."""
    raise NotImplementedError(
        "has_lerobot_metadata not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def prepare_lerobot_root_for_create(root: str | Path) -> Path | None:
    """Allow LeRobotDataset.create to use a clean local root."""
    raise NotImplementedError(
        "prepare_lerobot_root_for_create not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def build_lerobot_features(
    cameras: Mapping[str, Mapping[str, int]], fps: int
) -> dict[str, dict[str, Any]]:
    """Build LeRobotDataset features including wrist and overhead videos."""
    raise NotImplementedError(
        "build_lerobot_features not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def discover_camera_metadata(scene: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Return camera dimensions for required teleop cameras in an Isaac scene."""
    raise NotImplementedError(
        "discover_camera_metadata not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def ensure_required_cameras(cameras: Mapping[str, Any]) -> None:
    """Raise ValueError if any required teleop camera is missing from ``cameras``."""
    raise NotImplementedError(
        "ensure_required_cameras not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def has_scene_entity(scene: Mapping[str, Any], name: str) -> bool:
    """Return whether a dict-like or Isaac InteractiveScene has an entity."""
    raise NotImplementedError(
        "has_scene_entity not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def get_scene_entity(scene: Mapping[str, Any], name: str) -> Any:
    """Retrieve a named entity from a dict-like or Isaac InteractiveScene."""
    raise NotImplementedError(
        "get_scene_entity not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def collect_camera_buffers(scene: Mapping[str, Any]) -> dict[str, "np.ndarray"]:
    """Collect one RGB frame from each required teleop camera."""
    raise NotImplementedError(
        "collect_camera_buffers not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def read_robot_state(
    robot: Any, sim_joint_names: tuple[str, ...] | None = None
) -> "np.ndarray":
    """Read simulated SO-ARM101 joint positions in LeRobot SO101 order."""
    raise NotImplementedError(
        "read_robot_state not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def read_robot_proprio(
    robot: Any, sim_joint_names: tuple[str, ...] | None = None
) -> tuple["np.ndarray", "np.ndarray"]:
    """Read simulated SO-ARM101 joint position and velocity."""
    raise NotImplementedError(
        "read_robot_proprio not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


def ordered_action_to_numpy(action_targets: Any) -> "np.ndarray":
    """Convert ordered action targets (tensor or array-like) to float32 numpy."""
    raise NotImplementedError(
        "ordered_action_to_numpy not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
    )


class SafeSim2RealLeRobotRecorder:
    """Small LeRobotDataset recorder for interactive teleop episodes.

    SKELETON — instantiation raises NotImplementedError.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        task_name: str,
        cameras: Mapping[str, Mapping[str, int]],
        fps: int,
    ):
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )

    @property
    def recording(self) -> bool:
        """Whether an episode is currently being recorded."""
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder.recording not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )

    def init_dataset(self) -> None:
        """Initialise or reopen the LeRobotDataset on disk."""
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder.init_dataset not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )

    def start_episode(self) -> None:
        """Begin recording a new episode."""
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder.start_episode not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )

    def add_frame(
        self,
        action: "np.ndarray",
        observation: "np.ndarray | None" = None,
        camera_buffers: Mapping[str, "np.ndarray"] | None = None,
        qpos: "np.ndarray | None" = None,
        qvel: "np.ndarray | None" = None,
        timestamp: float | None = None,
    ) -> None:
        """Append a single teleop frame to the in-progress episode."""
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder.add_frame not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )

    def save_episode(self, success: bool = False) -> None:
        """Finalise and persist the current episode."""
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder.save_episode not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )

    def cancel_episode(self) -> None:
        """Discard the in-progress episode without saving."""
        raise NotImplementedError(
            "SafeSim2RealLeRobotRecorder.cancel_episode not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/lerobot_recorder.py"
        )


__all__ = [
    "REQUIRED_CAMERA_NAMES",
    "REQUIRED_REOPEN_METADATA",
    "has_lerobot_metadata",
    "prepare_lerobot_root_for_create",
    "build_lerobot_features",
    "discover_camera_metadata",
    "ensure_required_cameras",
    "has_scene_entity",
    "get_scene_entity",
    "collect_camera_buffers",
    "read_robot_state",
    "read_robot_proprio",
    "ordered_action_to_numpy",
    "SafeSim2RealLeRobotRecorder",
]
