# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Local HDF5 recording for OpenSO-101 teleoperation episodes.

The recorder streams chunked, resizable datasets to disk as frames arrive so a
crash mid-episode does not lose previously captured frames. Frames are
buffered in RAM up to ``flush_steps`` and then appended to the on-disk
datasets in a single write per field. Existing public API (class name,
constructor signature, ``start_episode`` / ``add_frame`` / ``save_episode`` /
``cancel_episode`` / ``create_checkpoint`` / ``restore_checkpoint``) is
preserved.
"""

from __future__ import annotations

import math
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
    "environment_origin",
    "scene_entity_states",
    "scene_jaw_forces",
    "scene_hold_seconds",
    "scene_success",
    "scene_program_phase",
    "scene_program_hold",
    "task_goal_root",
    "task_hold_seconds",
    "task_episode_step",
    "object_root_state",
    "command_stage",
    "command_goal_pos_b",
    "command_goal_pos_w",
    "command_cube_spawn_xy_b",
    "command_placement_hold_seconds",
    "command_pose_command_b",
    "cohort_environment_origins",
    "cohort_joint_position",
    "cohort_joint_velocity",
    "cohort_joint_targets",
    "cohort_policy_actions",
    "cohort_object_root_state",
    "cohort_command_stage",
    "cohort_command_goal_pos_b",
    "cohort_command_goal_pos_w",
    "cohort_command_cube_spawn_xy_b",
    "cohort_command_placement_hold_seconds",
    "cohort_command_pose_command_b",
    "cohort_task_hold_seconds",
    "cohort_task_episode_step",
)


def _episode_files(root: str | Path) -> list[Path]:
    episodes_dir = Path(root) / "episodes"
    if not episodes_dir.is_dir():
        return []
    # Ignore temporary/foreign files that happen to share the glob.  The
    # recorder's numbering contract is ``episode_<integer>.hdf5``; accepting
    # a malformed name here would make ``next_episode_path`` fail before a
    # user can recover the valid recordings.
    return sorted(
        path for path in episodes_dir.glob(HDF5_EPISODE_GLOB)
        if path.stem.rsplit("_", 1)[-1].isdigit()
    )


def validate_hdf5_episode(path: str | Path) -> None:
    """Validate the HDF5 layout used by local teleop recording."""

    with h5py.File(path, "r") as h5:
        for dataset_name in REQUIRED_HDF5_DATASETS:
            if dataset_name not in h5:
                raise ValueError(f"{path} is missing required dataset: {dataset_name}")
        action = h5["action"]
        if action.ndim != 2:
            raise ValueError(f"{path} action must be a rank-2 dataset (T, joints)")
        frame_count = action.shape[0]
        if frame_count <= 0:
            raise ValueError(f"{path} contains no frames")
        for dataset_name in REQUIRED_HDF5_DATASETS:
            if h5[dataset_name].shape[0] != frame_count:
                raise ValueError(
                    f"{path} has inconsistent frame count for {dataset_name}: "
                    f"{h5[dataset_name].shape[0]} != {frame_count}"
                )
        joint_shape = (len(SO101_TELEOP_CONTROL_JOINT_NAMES),)
        if action.shape[1:] != joint_shape:
            raise ValueError(f"{path} action shape must be (T, {len(SO101_TELEOP_CONTROL_JOINT_NAMES)})")
        for dataset_name in ("observations/qpos", "observations/qvel"):
            dataset = h5[dataset_name]
            if dataset.ndim != 2 or dataset.shape[1:] != joint_shape:
                raise ValueError(
                    f"{path} {dataset_name} shape must be (T, {len(SO101_TELEOP_CONTROL_JOINT_NAMES)})"
                )
            values = np.asarray(dataset[:], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{path} {dataset_name} contains non-finite values")
        action_values = np.asarray(action[:], dtype=np.float64)
        if not np.isfinite(action_values).all():
            raise ValueError(f"{path} action contains non-finite values")
        timestamps = h5["timestamps"]
        if timestamps.ndim != 1:
            raise ValueError(f"{path} timestamps must be a rank-1 dataset")
        timestamp_values = np.asarray(timestamps[:], dtype=np.float64)
        if not np.isfinite(timestamp_values).all():
            raise ValueError(f"{path} timestamps contains non-finite values")
        if "fps" in h5.attrs:
            fps = float(h5.attrs["fps"])
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{path} fps must be a positive finite number")
        for camera_name in REQUIRED_CAMERA_NAMES:
            dataset = h5[f"observations/images/{camera_name}"]
            if dataset.ndim != 4 or dataset.shape[1] <= 0 or dataset.shape[2] <= 0 or dataset.shape[3] != 3:
                raise ValueError(
                    f"{path} observations/images/{camera_name} must have shape (T, H, W, 3)"
                )
            if dataset.dtype.kind not in "uif":
                raise ValueError(f"{path} camera dataset {camera_name} must contain numeric pixels")
            if dataset.dtype.kind == "f" and not np.isfinite(dataset[:]).all():
                raise ValueError(f"{path} camera dataset {camera_name} contains non-finite pixels")
        if "sim" in h5:
            if "environment_origin" in h5["sim"] and h5["sim/environment_origin"].shape != (frame_count, 3):
                raise ValueError("environment_origin 必须为每个帧的三维环境原点")
            for name, dataset in h5["sim"].items():
                if dataset.shape[0] != frame_count:
                    raise ValueError(
                        f"{path} sim/{name} has inconsistent frame count: "
                        f"{dataset.shape[0]} != {frame_count}"
                    )
                if dataset.dtype.kind not in "biuf":
                    raise ValueError(f"{path} sim/{name} must contain numeric values")
                if dataset.dtype.kind == "f" and not np.isfinite(dataset[:]).all():
                    raise ValueError(f"{path} sim/{name} contains non-finite values")
        if "source_num_envs" in h5.attrs:
            count = int(h5.attrs["source_num_envs"])
            if count <= 0:
                raise ValueError("来源环境数量需要大于零")
            for field in ("source_seed", "source_env_spacing", "source_replicate_physics"):
                if field not in h5.attrs:
                    raise ValueError(f"并行采集缺少配置: {field}")
            shapes = {"environment_origins": (count, 3), "joint_position": (count, 6),
                      "joint_velocity": (count, 6), "joint_targets": (count, 6), "policy_actions": (count, 6),
                      "object_root_state": (count, 13), "task_episode_step": (count,)}
            for field, shape in shapes.items():
                key = f"sim/cohort_{field}"
                if key not in h5 or h5[key].shape != (frame_count, *shape):
                    raise ValueError(f"并行采集的字段形状错误: {key}")
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
        env_id: str | None = None,
        scene_metadata: Mapping[str, str] | None = None,
    ):
        self.root = Path(root)
        self.task_name = task_name
        # The gym env ID (e.g. "OpenSO101-Stack-v0") that recorded this
        # episode. Replay reads it from the HDF5 attrs to spawn the same
        # scene; without it, replay can only guess and may render the
        # wrong task (default PickPlace).
        self.env_id = env_id
        self.scene_metadata = dict(scene_metadata or {})
        if set(self.scene_metadata) - {"scene_sha256", "scene_relative_path"}:
            raise ValueError("scene_metadata 包含未知字段")
        self.cameras = dict(cameras)
        self.fps = fps
        self.dataset_id = dataset_id or "local/openso101_pickplace_teleop"
        self.sim_joint_names = tuple(sim_joint_names or SO101_SIM_JOINT_NAMES)
        ensure_required_cameras(self.cameras)
        if int(fps) <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if len(self.sim_joint_names) != len(SO101_TELEOP_CONTROL_JOINT_NAMES):
            raise ValueError(
                "sim_joint_names must contain exactly "
                f"{len(SO101_TELEOP_CONTROL_JOINT_NAMES)} joints"
            )
        for camera_name in REQUIRED_CAMERA_NAMES:
            camera = self.cameras[camera_name]
            try:
                height, width = int(camera["height"]), int(camera["width"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"camera {camera_name!r} needs integer height and width") from exc
            if height <= 0 or width <= 0:
                raise ValueError(f"camera {camera_name!r} dimensions must be positive")
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
        # 首帧确定 sim state 字段，后续每帧保持一致。
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
        if self.env_id is not None:
            h5.attrs["env_id"] = self.env_id
        for key, value in self.scene_metadata.items():
            h5.attrs[key] = value
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

        if int(checkpoint) != checkpoint:
            raise ValueError(f"checkpoint must be an integer, got {checkpoint!r}")
        target = int(checkpoint)
        if target < 0 or target > self.total_frames:
            raise ValueError(
                f"checkpoint {target} is outside the current episode range [0, {self.total_frames}]"
            )
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
        joint_shape = (len(SO101_TELEOP_CONTROL_JOINT_NAMES),)
        frame_arrays = {
            "action": np.asarray(action),
            "qpos": np.asarray(qpos),
            "qvel": np.asarray(qvel),
        }
        for name, value in frame_arrays.items():
            if value.shape != joint_shape:
                raise ValueError(f"{name} must have shape {joint_shape}, got {value.shape}")
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite numeric values")
        if not math.isfinite(float(timestamp)):
            raise ValueError("timestamp must be finite")
        for camera_name in REQUIRED_CAMERA_NAMES:
            image = np.asarray(camera_buffers[camera_name])
            expected = (
                int(self.cameras[camera_name]["height"]),
                int(self.cameras[camera_name]["width"]),
                3,
            )
            if image.shape != expected:
                raise ValueError(f"camera {camera_name!r} must have shape {expected}, got {image.shape}")
            if image.dtype.kind not in "uif" or (image.dtype.kind == "f" and not np.isfinite(image).all()):
                raise ValueError(f"camera {camera_name!r} must contain finite numeric pixels")
        frame_sim: dict[str, np.ndarray] = {}
        if sim_state:
            unknown = set(sim_state) - set(SIM_STATE_KEYS)
            if unknown:
                raise ValueError(f"未知 sim state 字段: {sorted(unknown)}")
            frame_sim = {
                key: np.asarray(value)
                for key, value in sim_state.items()
            }
            for key, value in frame_sim.items():
                if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                    raise ValueError(f"sim state {key!r} must contain finite numeric values")
            if self._sim_keys is not None:
                for key in self._sim_keys:
                    if key in frame_sim:
                        expected_shape = self._sim_shape(key)
                        if frame_sim[key].shape != expected_shape:
                            raise ValueError(
                                f"sim state {key!r} shape changed from {expected_shape} "
                                f"to {frame_sim[key].shape}"
                            )
        if self.scene_metadata and "scene_entity_states" not in frame_sim:
            raise ValueError("自定义场景的每个采集帧都必须包含全部实体状态")
        if self._sim_keys is not None and set(frame_sim) != set(self._sim_keys):
            raise ValueError("每个采集帧的 sim state 字段必须与首帧一致")
        if self._sim_keys is None:
            self._sim_keys = tuple(key for key in SIM_STATE_KEYS if key in frame_sim)
            if self._sim_keys:
                self._create_sim_datasets(frame_sim)
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

    def _sim_shape(self, key: str) -> tuple[int, ...]:
        if self._h5 is None or "sim" not in self._h5 or key not in self._h5["sim"]:
            raise KeyError(key)
        return tuple(self._h5["sim"][key].shape[1:])

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

        if self._sim_keys:
            sim = h5["sim"]
            for key in self._sim_keys:
                ds = sim[key]
                arr = np.stack([frame["sim_state"][key] for frame in buffer], axis=0)
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
