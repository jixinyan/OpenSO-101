# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Local HDF5 recording for Safe Sim2Real teleoperation episodes.

The recorder streams chunked, resizable datasets to disk as frames arrive so a
crash mid-episode does not lose previously captured frames. Frames are
buffered in RAM up to ``flush_steps`` and then appended to the on-disk
datasets in a single write per field. Existing public API (class name,
constructor signature, ``start_episode`` / ``add_frame`` / ``save_episode`` /
``cancel_episode`` / ``create_checkpoint`` / ``restore_checkpoint``) is
preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from .lerobot_recorder import REQUIRED_CAMERA_NAMES, ensure_required_cameras
from openso101.robots import SO101_SIM_JOINT_NAMES

from .so101_mapping import LEROBOT_SO101_ACTION_NAMES, SO101_TELEOP_CONTROL_JOINT_NAMES

HDF5_EPISODE_GLOB = "episode_*.hdf5"
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


def _episode_files(root: str | Path) -> list[Path]:
    episodes_dir = Path(root) / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(episodes_dir.glob(HDF5_EPISODE_GLOB))


def validate_hdf5_episode(path: str | Path) -> None:
    """Validate the HDF5 layout used by local teleop recording."""

    with h5py.File(path, "r") as h5:
        for dataset_name in REQUIRED_HDF5_DATASETS:
            if dataset_name not in h5:
                raise ValueError(f"{path} is missing required dataset: {dataset_name}")
        frame_count = h5["action"].shape[0]
        for dataset_name in REQUIRED_HDF5_DATASETS:
            if h5[dataset_name].shape[0] != frame_count:
                raise ValueError(
                    f"{path} has inconsistent frame count for {dataset_name}: "
                    f"{h5[dataset_name].shape[0]} != {frame_count}"
                )
        if h5["action"].shape[1:] != (len(SO101_TELEOP_CONTROL_JOINT_NAMES),):
            raise ValueError(f"{path} action shape must be (T, {len(SO101_TELEOP_CONTROL_JOINT_NAMES)})")
        if h5["observations/qpos"].shape[1:] != (len(SO101_TELEOP_CONTROL_JOINT_NAMES),):
            raise ValueError(f"{path} observations/qpos shape must be (T, {len(SO101_TELEOP_CONTROL_JOINT_NAMES)})")
        if h5["observations/qvel"].shape[1:] != (len(SO101_TELEOP_CONTROL_JOINT_NAMES),):
            raise ValueError(f"{path} observations/qvel shape must be (T, {len(SO101_TELEOP_CONTROL_JOINT_NAMES)})")


def validate_hdf5_dataset(root: str | Path) -> list[Path]:
    """Return valid HDF5 episode files, or raise a useful validation error."""

    episode_files = _episode_files(root)
    if not episode_files:
        raise ValueError(f"Local HDF5 teleop dataset has no episodes under {Path(root) / 'episodes'}.")
    for episode_file in episode_files:
        validate_hdf5_episode(episode_file)
    return episode_files


class OpenSO101HDF5TeleopRecorder:
    """Streaming chunked HDF5 recorder with an ACT/LeRobot-friendly layout.

    Frames added via :meth:`add_frame` are buffered and flushed to disk in
    chunks of ``flush_steps`` so a crash mid-episode preserves all flushed
    frames. ``flush_steps``, ``chunks_length``, and ``compression`` are
    constructor kwargs with sensible defaults so existing callers are
    unaffected.
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
        self.root = Path(root)
        self.task_name = task_name
        self.cameras = dict(cameras)
        self.fps = fps
        self.dataset_id = dataset_id or "local/safe_sim2real_pickplace_teleop"
        self.sim_joint_names = tuple(sim_joint_names or SO101_SIM_JOINT_NAMES)
        ensure_required_cameras(self.cameras)
        if flush_steps < 1:
            raise ValueError(f"flush_steps must be >= 1, got {flush_steps}")
        if chunks_length < 1:
            raise ValueError(f"chunks_length must be >= 1, got {chunks_length}")
        self.flush_steps = int(flush_steps)
        self.chunks_length = int(chunks_length)
        self.compression = compression

        self._recording = False
        # In-memory tail buffer: list of per-frame dicts not yet flushed.
        self._buffer: list[dict[str, Any]] = []
        # Total frames already flushed to disk (excludes buffered frames).
        self._flushed_frames = 0
        # Checkpoint frame indices recorded for the current episode.
        self._checkpoints: list[int] = []
        # Sim-state key set discovered on the first frame that supplied
        # sim_state. Subsequent frames are expected to provide the same keys.
        self._sim_keys: tuple[str, ...] | None = None
        # Current on-disk file handle and path for the in-progress episode.
        self._h5: h5py.File | None = None
        self._episode_path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def total_frames(self) -> int:
        """Total frames captured in the current episode (flushed + buffered)."""

        return self._flushed_frames + len(self._buffer)

    def init_dataset(self) -> None:
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        print(f"[INFO]: Local HDF5 teleop dataset root: {self.root}")

    def next_episode_path(self) -> Path:
        episodes_dir = self.root / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        existing = _episode_files(self.root)
        if not existing:
            return episodes_dir / "episode_000000.hdf5"
        last_index = max(int(path.stem.rsplit("_", 1)[-1]) for path in existing)
        return episodes_dir / f"episode_{last_index + 1:06d}.hdf5"

    def current_episode_path(self) -> Path | None:
        """Path of the on-disk file for the in-progress episode, if any."""

        return self._episode_path

    def start_episode(self) -> None:
        self.init_dataset()
        # Clean up any leftover state from a previous incomplete episode.
        self._close_file()
        self._recording = True
        self._buffer = []
        self._flushed_frames = 0
        self._checkpoints = []
        self._sim_keys = None
        self._episode_path = self.next_episode_path()
        self._h5 = h5py.File(self._episode_path, "w")
        self._write_episode_attrs(success=False)
        self._create_core_datasets()
        print(f"[INFO]: Started local HDF5 streaming recording: {self._episode_path}")

    def _write_episode_attrs(self, *, success: bool) -> None:
        assert self._h5 is not None
        h5 = self._h5
        h5.attrs["format"] = "openso101_teleop_hdf5_v1"
        h5.attrs["dataset_id"] = self.dataset_id
        h5.attrs["task"] = self.task_name
        h5.attrs["fps"] = int(self.fps)
        h5.attrs["success"] = bool(success)
        h5.attrs["joint_names"] = np.asarray(SO101_TELEOP_CONTROL_JOINT_NAMES, dtype=h5py.string_dtype())
        h5.attrs["sim_joint_names"] = np.asarray(self.sim_joint_names, dtype=h5py.string_dtype())
        h5.attrs["lerobot_action_names"] = np.asarray(LEROBOT_SO101_ACTION_NAMES, dtype=h5py.string_dtype())
        h5.attrs["camera_names"] = np.asarray(REQUIRED_CAMERA_NAMES, dtype=h5py.string_dtype())

    def _create_streaming_dataset(
        self,
        parent: h5py.Group,
        name: str,
        feature_shape: tuple[int, ...],
        dtype: Any,
        *,
        chunks_first_dim: int | None = None,
    ) -> h5py.Dataset:
        """Create a resizable chunked dataset for streaming writes."""

        chunk_rows = int(chunks_first_dim if chunks_first_dim is not None else self.chunks_length)
        chunk_rows = max(chunk_rows, 1)
        chunks = (chunk_rows, *feature_shape)
        kwargs: dict[str, Any] = {
            "shape": (0, *feature_shape),
            "maxshape": (None, *feature_shape),
            "dtype": dtype,
            "chunks": chunks,
        }
        if self.compression is not None:
            kwargs["compression"] = self.compression
        return parent.create_dataset(name, **kwargs)

    def _create_core_datasets(self) -> None:
        assert self._h5 is not None
        h5 = self._h5
        joint_dim = len(SO101_TELEOP_CONTROL_JOINT_NAMES)
        self._create_streaming_dataset(h5, "action", (joint_dim,), np.float32)
        # timestamps is per-frame scalar
        h5.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(max(self.chunks_length, 1),),
        )
        checkpoints = h5.create_group("checkpoints")
        checkpoints.create_dataset(
            "frame_index",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(max(min(self.chunks_length, 16), 1),),
        )
        observations = h5.create_group("observations")
        self._create_streaming_dataset(observations, "qpos", (joint_dim,), np.float32)
        self._create_streaming_dataset(observations, "qvel", (joint_dim,), np.float32)
        images = observations.create_group("images")
        for camera_name in REQUIRED_CAMERA_NAMES:
            height = int(self.cameras[camera_name]["height"])
            width = int(self.cameras[camera_name]["width"])
            self._create_streaming_dataset(
                images,
                camera_name,
                (height, width, 3),
                np.uint8,
                chunks_first_dim=1,
            )

    def create_checkpoint(self) -> int:
        """Return a restore point for the current episode."""

        checkpoint = self.total_frames
        replay_frame = max(checkpoint - 1, 0)
        if not self._checkpoints or self._checkpoints[-1] != replay_frame:
            self._checkpoints.append(replay_frame)
        return checkpoint

    def restore_checkpoint(self, checkpoint: int) -> None:
        """Truncate buffered and on-disk frames back to ``checkpoint``."""

        target = max(int(checkpoint), 0)
        # First drop buffered frames beyond the target.
        if target <= self._flushed_frames:
            # Need to truncate the on-disk datasets as well.
            self._buffer = []
            self._truncate_on_disk(target)
            self._flushed_frames = target
        else:
            keep_in_buffer = target - self._flushed_frames
            self._buffer = self._buffer[:keep_in_buffer]
        self._checkpoints = [idx for idx in self._checkpoints if idx < max(target, 1)]
        self._recording = True
        print(f"[INFO]: Restored local HDF5 recording to checkpoint frame {target}.")

    def _truncate_on_disk(self, new_length: int) -> None:
        """Shrink all streaming datasets to ``new_length`` rows."""

        if self._h5 is None:
            return
        h5 = self._h5
        for path in (
            "action",
            "timestamps",
            "observations/qpos",
            "observations/qvel",
        ):
            ds = h5[path]
            ds.resize((new_length, *ds.shape[1:]))
        images = h5["observations/images"]
        for camera_name in REQUIRED_CAMERA_NAMES:
            ds = images[camera_name]
            ds.resize((new_length, *ds.shape[1:]))
        if self._sim_keys is not None and "sim" in h5:
            sim = h5["sim"]
            for key in self._sim_keys:
                if key in sim:
                    ds = sim[key]
                    ds.resize((new_length, *ds.shape[1:]))
        h5.flush()

    def add_frame(
        self,
        action: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        camera_buffers: Mapping[str, np.ndarray],
        timestamp: float,
        sim_state: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._recording:
            return
        ensure_required_cameras(camera_buffers)
        frame_sim: dict[str, np.ndarray] = {}
        if sim_state:
            frame_sim = {
                key: np.asarray(value)
                for key, value in sim_state.items()
                if key in SIM_STATE_KEYS and value is not None
            }
        # First frame with sim_state pins the schema; lazily create datasets.
        if frame_sim and self._sim_keys is None:
            self._sim_keys = tuple(key for key in SIM_STATE_KEYS if key in frame_sim)
            self._create_sim_datasets(frame_sim)
            # Back-fill any frames already flushed without sim_state with
            # zero-valued rows so dataset lengths stay consistent. In normal
            # callers the very first frame either provides sim_state or no
            # frame does, so this is a defensive edge case.
            if self._flushed_frames > 0 and self._h5 is not None:
                sim = self._h5["sim"]
                for key in self._sim_keys:
                    ds = sim[key]
                    ds.resize((self._flushed_frames, *ds.shape[1:]))
        self._buffer.append(
            {
                "action": np.asarray(action, dtype=np.float32),
                "qpos": np.asarray(qpos, dtype=np.float32),
                "qvel": np.asarray(qvel, dtype=np.float32),
                "timestamp": float(timestamp),
                "camera_buffers": {
                    camera_name: np.asarray(camera_buffers[camera_name], dtype=np.uint8)
                    for camera_name in REQUIRED_CAMERA_NAMES
                },
                "sim_state": frame_sim,
            }
        )
        if len(self._buffer) >= self.flush_steps:
            self.flush()

    def _create_sim_datasets(self, first_sim_state: Mapping[str, np.ndarray]) -> None:
        assert self._h5 is not None
        h5 = self._h5
        sim = h5.require_group("sim")
        for key in self._sim_keys or ():
            sample = np.asarray(first_sim_state[key])
            feature_shape = tuple(sample.shape)
            chunk_rows = max(self.chunks_length, 1)
            kwargs: dict[str, Any] = {
                "shape": (0, *feature_shape),
                "maxshape": (None, *feature_shape),
                "dtype": sample.dtype,
                "chunks": (chunk_rows, *feature_shape) if feature_shape else (chunk_rows,),
            }
            if self.compression is not None:
                kwargs["compression"] = self.compression
            sim.create_dataset(key, **kwargs)

    def flush(self) -> None:
        """Persist any buffered frames to disk."""

        if self._h5 is None or not self._buffer:
            return
        h5 = self._h5
        buffer = self._buffer
        n = len(buffer)
        start = self._flushed_frames
        end = start + n

        action_arr = np.stack([frame["action"] for frame in buffer], axis=0)
        qpos_arr = np.stack([frame["qpos"] for frame in buffer], axis=0)
        qvel_arr = np.stack([frame["qvel"] for frame in buffer], axis=0)
        timestamps_arr = np.asarray([frame["timestamp"] for frame in buffer], dtype=np.float64)

        for path, arr in (
            ("action", action_arr),
            ("observations/qpos", qpos_arr),
            ("observations/qvel", qvel_arr),
        ):
            ds = h5[path]
            ds.resize((end, *ds.shape[1:]))
            ds[start:end] = arr
        ts = h5["timestamps"]
        ts.resize((end,))
        ts[start:end] = timestamps_arr

        images = h5["observations/images"]
        for camera_name in REQUIRED_CAMERA_NAMES:
            ds = images[camera_name]
            cam_arr = np.stack(
                [frame["camera_buffers"][camera_name] for frame in buffer], axis=0
            )
            ds.resize((end, *ds.shape[1:]))
            ds[start:end] = cam_arr

        if self._sim_keys is not None:
            sim = h5["sim"]
            for key in self._sim_keys:
                ds = sim[key]
                samples = []
                for frame in buffer:
                    if key in frame["sim_state"]:
                        samples.append(np.asarray(frame["sim_state"][key]))
                    else:
                        # Missing sim_state for an established key: zero-fill
                        # to keep length consistent. This preserves the
                        # implicit "all frames have sim_state" contract that
                        # the previous batched recorder relied on.
                        samples.append(np.zeros(ds.shape[1:], dtype=ds.dtype))
                arr = np.stack(samples, axis=0) if samples else np.empty((0, *ds.shape[1:]), dtype=ds.dtype)
                ds.resize((end, *ds.shape[1:]))
                ds[start:end] = arr

        self._flushed_frames = end
        self._buffer = []
        h5.flush()

    def save_episode(self, success: bool = False) -> Path | None:
        if not self._recording:
            return None
        self._recording = False
        if self.total_frames == 0:
            self._close_file()
            # Remove the empty file we created on start_episode.
            if self._episode_path is not None and self._episode_path.exists():
                self._episode_path.unlink()
            self._episode_path = None
            print("[WARN]: No frames recorded; skipping empty HDF5 episode.")
            return None

        self.flush()
        assert self._h5 is not None
        h5 = self._h5
        # Finalize attrs and checkpoints group.
        h5.attrs["success"] = bool(success)
        checkpoints_ds = h5["checkpoints/frame_index"]
        checkpoints_ds.resize((len(self._checkpoints),))
        if self._checkpoints:
            checkpoints_ds[:] = np.asarray(self._checkpoints, dtype=np.int64)

        frame_count = self._flushed_frames
        episode_path = self._episode_path
        self._close_file()
        self._buffer = []
        self._flushed_frames = 0
        self._checkpoints = []
        self._sim_keys = None
        self._episode_path = None
        print(f"[INFO]: Saved local HDF5 episode with {frame_count} frames: {episode_path}")
        return episode_path

    def cancel_episode(self) -> None:
        if not self._recording:
            return
        self._recording = False
        episode_path = self._episode_path
        self._close_file()
        if episode_path is not None and episode_path.exists():
            try:
                episode_path.unlink()
            except OSError:
                pass
        self._buffer = []
        self._flushed_frames = 0
        self._checkpoints = []
        self._sim_keys = None
        self._episode_path = None
        print("[INFO]: Cancelled local HDF5 episode.")

    def _close_file(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
