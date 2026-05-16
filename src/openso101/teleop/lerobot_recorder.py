# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""LeRobot dataset recording helpers for Safe Sim2Real teleoperation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from openso101.robots import SO101_SIM_JOINT_NAMES

from .so101_mapping import (
    LEROBOT_SO101_ACTION_NAMES,
    SO101_TELEOP_CONTROL_JOINT_NAMES,
    batched_action_to_motor_units,
)

REQUIRED_CAMERA_NAMES: tuple[str, str] = ("wrist_camera", "overhead_camera")
REQUIRED_REOPEN_METADATA: tuple[Path, ...] = (Path("meta/info.json"), Path("meta/tasks.parquet"))


def has_lerobot_metadata(root: str | Path) -> bool:
    """Return whether ``root`` has enough local metadata for LeRobot to reopen."""

    root_path = Path(root)
    return all((root_path / relative_path).is_file() for relative_path in REQUIRED_REOPEN_METADATA)


def _archive_incomplete_lerobot_root(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = root.with_name(f"{root.name}.incomplete-{timestamp}")
    suffix = 1
    while archive.exists():
        archive = root.with_name(f"{root.name}.incomplete-{timestamp}-{suffix}")
        suffix += 1
    root.rename(archive)
    return archive


def prepare_lerobot_root_for_create(root: str | Path) -> Path | None:
    """Allow LeRobotDataset.create to use a clean local root.

    LeRobotDataset.create expects the dataset root not to exist. A previous
    aborted first run can leave an empty directory or incomplete metadata
    behind. Empty directories are removed. LeRobot-looking incomplete roots are
    archived so no collected files are deleted.
    """

    root_path = Path(root)
    if not root_path.exists():
        return None
    if has_lerobot_metadata(root_path):
        return None
    if (root_path / "meta" / "info.json").is_file():
        return _archive_incomplete_lerobot_root(root_path)
    try:
        root_path.rmdir()
    except OSError as exc:
        raise ValueError(
            f"Dataset root exists but is not a LeRobot dataset and is not empty: {root_path}. "
            "Choose a new --repo-root or remove the stale directory after checking its contents."
        ) from exc
    return None


def build_lerobot_features(cameras: Mapping[str, Mapping[str, int]], fps: int) -> dict[str, dict[str, Any]]:
    """Build LeRobotDataset features including wrist and overhead videos."""

    ensure_required_cameras(cameras)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "fps": fps,
            "shape": (len(SO101_TELEOP_CONTROL_JOINT_NAMES),),
            "names": list(LEROBOT_SO101_ACTION_NAMES),
        },
        "action": {
            "dtype": "float32",
            "fps": fps,
            "shape": (len(LEROBOT_SO101_ACTION_NAMES),),
            "names": list(LEROBOT_SO101_ACTION_NAMES),
        },
    }

    for camera_name in REQUIRED_CAMERA_NAMES:
        camera = cameras[camera_name]
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "fps": fps,
            "shape": (int(camera["height"]), int(camera["width"]), 3),
            "names": ["height", "width", "channels"],
        }
    return features


def discover_camera_metadata(scene: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Return camera dimensions for required teleop cameras in an Isaac scene."""

    cameras: dict[str, dict[str, int]] = {}
    for camera_name in REQUIRED_CAMERA_NAMES:
        if not has_scene_entity(scene, camera_name):
            continue
        sensor = get_scene_entity(scene, camera_name)
        cameras[camera_name] = {
            "height": int(sensor.cfg.height),
            "width": int(sensor.cfg.width),
        }
    ensure_required_cameras(cameras)
    return cameras


def ensure_required_cameras(cameras: Mapping[str, Any]) -> None:
    missing = [camera_name for camera_name in REQUIRED_CAMERA_NAMES if not has_scene_entity(cameras, camera_name)]
    if missing:
        raise ValueError(
            "Teleop data collection requires camera-enabled tasks with "
            f"{', '.join(REQUIRED_CAMERA_NAMES)}. Missing: {', '.join(missing)}"
        )


def has_scene_entity(scene: Mapping[str, Any], name: str) -> bool:
    """Return whether a dict-like or Isaac InteractiveScene has an entity."""

    if hasattr(scene, "keys"):
        return name in set(scene.keys())
    try:
        scene[name]
    except KeyError:
        return False
    else:
        return True


def get_scene_entity(scene: Mapping[str, Any], name: str) -> Any:
    return scene[name]


def _as_numpy_rgb(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0.0, 255.0)
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def collect_camera_buffers(scene: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Collect one RGB frame from each required teleop camera."""

    ensure_required_cameras(scene)
    return {
        camera_name: _as_numpy_rgb(get_scene_entity(scene, camera_name).data.output["rgb"])
        for camera_name in REQUIRED_CAMERA_NAMES
    }


def read_robot_state(robot: Any, sim_joint_names: tuple[str, ...] | None = None) -> np.ndarray:
    """Read simulated SO-ARM101 joint positions in LeRobot SO101 order."""

    joint_names = list(robot.joint_names)
    requested_joint_names = sim_joint_names or SO101_SIM_JOINT_NAMES
    indices = [joint_names.index(joint_name) for joint_name in requested_joint_names]
    joint_pos = robot.data.joint_pos[0, indices]
    if hasattr(joint_pos, "detach"):
        joint_pos = joint_pos.detach().cpu().numpy()
    return np.asarray(joint_pos, dtype=np.float32)


def read_robot_proprio(robot: Any, sim_joint_names: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read simulated SO-ARM101 joint position and velocity."""

    joint_names = list(robot.joint_names)
    requested_joint_names = sim_joint_names or SO101_SIM_JOINT_NAMES
    indices = [joint_names.index(joint_name) for joint_name in requested_joint_names]
    joint_pos = robot.data.joint_pos[0, indices]
    joint_vel = robot.data.joint_vel[0, indices]
    if hasattr(joint_pos, "detach"):
        joint_pos = joint_pos.detach().cpu().numpy()
    if hasattr(joint_vel, "detach"):
        joint_vel = joint_vel.detach().cpu().numpy()
    return np.asarray(joint_pos, dtype=np.float32), np.asarray(joint_vel, dtype=np.float32)


def ordered_action_to_numpy(action_targets: Any) -> np.ndarray:
    if hasattr(action_targets, "detach"):
        action_targets = action_targets.detach().cpu().numpy()
    return np.asarray(action_targets, dtype=np.float32)


def _sim_radians_array_to_motor_units(values: Any) -> np.ndarray:
    """Remap a 6-vector (or batched) of sim-radian joint values to motor units.

    Accepts numpy arrays, torch tensors, or sequences. Returns float32 numpy
    so the downstream LeRobot writer can serialize it without further conversion.
    The remap matches the convention real STS3215 hardware reports/accepts.
    """
    import torch  # local: avoid mandatory torch import at module load

    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(torch.float32).cpu()
    else:
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
    motor = batched_action_to_motor_units(tensor)
    return motor.numpy().astype(np.float32, copy=False)


class OpenSO101LeRobotRecorder:
    """Small LeRobotDataset recorder for interactive teleop episodes."""

    def __init__(self, repo_id: str, root: str, task_name: str, cameras: Mapping[str, Mapping[str, int]], fps: int):
        self.repo_id = repo_id
        self.root = root
        self.task_name = task_name
        self.cameras = dict(cameras)
        self.fps = fps
        self.features = build_lerobot_features(self.cameras, fps=fps)
        self._dataset = None
        self._recording = False
        self._frames_in_episode = 0

    @property
    def recording(self) -> bool:
        return self._recording

    def init_dataset(self) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is required for dataset recording. Install it via "
                "`bash scripts/install.sh` from the repo root (or, manually, "
                "`pip install \"lerobot[feetech]==0.4.0\" --no-deps` to bypass "
                "the isaaclab/lerobot packaging version conflict)."
            ) from exc

        if has_lerobot_metadata(self.root):
            self._dataset = LeRobotDataset(self.repo_id, root=self.root)
            print(f"[INFO]: Existing LeRobot dataset opened at {self.root}")
            return

        archived_root = prepare_lerobot_root_for_create(self.root)
        if archived_root is not None:
            print(
                "[WARN]: Archived incomplete local LeRobot dataset root "
                f"from {self.root} to {archived_root}. A fresh local dataset will be created."
            )
        self._dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self.features,
            root=self.root,
            robot_type="so101_follower",
        )
        print(f"[INFO]: New LeRobot dataset created at {self.root}")

    def start_episode(self) -> None:
        if self._dataset is None:
            self.init_dataset()
        self._recording = True
        self._frames_in_episode = 0
        print("[INFO]: Started LeRobot recording.")

    def add_frame(
        self,
        action: np.ndarray,
        observation: np.ndarray | None = None,
        camera_buffers: Mapping[str, np.ndarray] | None = None,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        timestamp: float | None = None,
        sim_state: Mapping[str, "np.ndarray"] | None = None,
    ) -> None:
        # ``sim_state`` is accepted for signature parity with
        # :class:`OpenSO101HDF5TeleopRecorder.add_frame` but is silently
        # dropped — the LeRobot dataset schema has no per-frame slot for
        # arbitrary sim state and recording it inline would invalidate
        # the upstream feature spec the policy trainer expects.
        del sim_state  # noqa: F841 — intentionally unused
        if not self._recording:
            return
        if observation is None:
            observation = qpos
        if observation is None:
            raise ValueError("LeRobot frame recording requires observation or qpos.")
        if camera_buffers is None:
            raise ValueError("LeRobot frame recording requires camera buffers.")
        ensure_required_cameras(camera_buffers)
        # Remap sim-radian action/state to LeRobot STS3215 motor units [-100, 100].
        # See openso101.teleop.so101_mapping for the per-joint linear map.
        # Camera frames are not remapped.
        action_motor = _sim_radians_array_to_motor_units(action)
        observation_motor = _sim_radians_array_to_motor_units(observation)
        frame = {
            "action": action_motor,
            "observation.state": observation_motor,
            "task": self.task_name,
        }
        for camera_name in REQUIRED_CAMERA_NAMES:
            frame[f"observation.images.{camera_name}"] = camera_buffers[camera_name]
        self._dataset.add_frame(frame)
        self._frames_in_episode += 1

    def save_episode(self, success: bool = False) -> None:
        if not self._recording:
            return
        self._recording = False
        if self._frames_in_episode == 0:
            print("[WARN]: No frames recorded; skipping empty episode.")
            return
        self._dataset.save_episode()
        print(f"[INFO]: Saved LeRobot episode with {self._frames_in_episode} frames.")

    def cancel_episode(self) -> None:
        if not self._recording:
            return
        self._recording = False
        if hasattr(self._dataset, "clear_episode_buffer"):
            self._dataset.clear_episode_buffer()
        print("[INFO]: Cancelled LeRobot episode.")

    # ----- Checkpoint API parity with OpenSO101HDF5TeleopRecorder -----
    #
    # LeRobotDataset doesn't have a native frame-level checkpoint
    # facility, so these are intentional no-ops that return ``None`` /
    # accept ``None``. The teleop checkpoint store still captures the
    # sim state (robot pose + object pose + command stage), and the R
    # key still restores from that — the only difference vs. HDF5
    # recording is that the LeRobot episode itself isn't rewound. If
    # you need replayable checkpoints, use ``--record-format hdf5``.

    def create_checkpoint(self):
        """Return a sentinel handle the caller can later pass to
        :meth:`restore_checkpoint`. The LeRobot recorder has no real
        rewind concept; this exists purely so the teleop dispatcher's
        capture path doesn't crash."""
        if self._recording:
            return self._frames_in_episode
        return None

    def restore_checkpoint(self, checkpoint) -> None:
        """No-op; see :meth:`create_checkpoint`. The sim state restore
        is handled separately by :class:`_TeleopCheckpointStore` in
        ``openso101.cli.il``."""
        del checkpoint  # nothing to do
        return None
