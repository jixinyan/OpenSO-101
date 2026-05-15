# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Local HDF5 recording for SO-101 teleoperation episodes.

SKELETON: bodies raise NotImplementedError until the real port from
`/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py` lands. The
legacy implementation is under active revision (motor-unit remap,
streaming HDF5 recording); this skeleton freezes the public surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


HDF5_EPISODE_GLOB: str = "episode_*.hdf5"

REQUIRED_HDF5_DATASETS: tuple[str, ...] = (
    "action",
    "observations/qpos",
    "observations/qvel",
    "observations/images/wrist_camera",
    "observations/images/overhead_camera",
    "timestamps",
)

SIM_STATE_KEYS: tuple[str, ...] = (
    "object_root_state",
    "command_stage",
    "command_goal_pos_b",
    "command_goal_pos_w",
    "command_cube_spawn_xy_b",
)


def validate_hdf5_episode(path: str | Path) -> None:
    """Validate the HDF5 layout used by local teleop recording."""
    raise NotImplementedError(
        "validate_hdf5_episode not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
    )


def validate_hdf5_dataset(root: str | Path) -> list[Path]:
    """Return valid HDF5 episode files, or raise a useful validation error."""
    raise NotImplementedError(
        "validate_hdf5_dataset not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
    )


class SafeSim2RealHDF5TeleopRecorder:
    """Streaming chunked HDF5 recorder with an ACT/LeRobot-friendly layout.

    SKELETON — instantiation raises NotImplementedError.
    """

    def __init__(
        self,
        root: str | Path,
        task_name: str,
        cameras: Mapping[str, Mapping[str, int]],
        fps: int,
        dataset_id: str | None = None,
        sim_joint_names: tuple[str, ...] | None = None,
        flush_steps: int = 100,
        chunks_length: int = 100,
        compression: str | None = "lzf",
    ):
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    @property
    def recording(self) -> bool:
        """Whether an episode is currently being recorded."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.recording not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    @property
    def total_frames(self) -> int:
        """Total frames captured in the current episode (flushed + buffered)."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.total_frames not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def init_dataset(self) -> None:
        """Create the episodes directory and print dataset root."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.init_dataset not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def next_episode_path(self) -> Path:
        """Return the path for the next episode file."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.next_episode_path not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def current_episode_path(self) -> Path | None:
        """Path of the on-disk file for the in-progress episode, if any."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.current_episode_path not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def start_episode(self) -> None:
        """Open a new HDF5 file and begin streaming recording."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.start_episode not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def create_checkpoint(self) -> int:
        """Return a restore point for the current episode."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.create_checkpoint not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def restore_checkpoint(self, checkpoint: int) -> None:
        """Truncate buffered and on-disk frames back to ``checkpoint``."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.restore_checkpoint not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def add_frame(
        self,
        action: "np.ndarray",
        qpos: "np.ndarray",
        qvel: "np.ndarray",
        camera_buffers: Mapping[str, "np.ndarray"],
        timestamp: float,
        sim_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Buffer a single teleop frame and flush to disk when the buffer is full."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.add_frame not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def flush(self) -> None:
        """Persist any buffered frames to disk."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.flush not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def save_episode(self, success: bool = False) -> Path | None:
        """Finalise and close the current HDF5 episode file."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.save_episode not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )

    def cancel_episode(self) -> None:
        """Discard and delete the in-progress HDF5 episode."""
        raise NotImplementedError(
            "SafeSim2RealHDF5TeleopRecorder.cancel_episode not yet ported. Source reference: "
            "/data/safe_sim2real/src/safe_sim2real/teleop/hdf5_recorder.py"
        )


__all__ = [
    "HDF5_EPISODE_GLOB",
    "REQUIRED_HDF5_DATASETS",
    "SIM_STATE_KEYS",
    "validate_hdf5_episode",
    "validate_hdf5_dataset",
    "SafeSim2RealHDF5TeleopRecorder",
]
