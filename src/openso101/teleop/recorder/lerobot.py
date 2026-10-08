# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from openso101.robots import SO101_SIM_JOINT_NAMES

from ..so101_mapping import (
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
    if isinstance(fps, bool) or not isinstance(fps, Integral) or fps <= 0:
        raise ValueError("fps 必须为正整数")
    fps = int(fps)
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
        height, width = camera["height"], camera["width"]
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 16
               for value in (height, width)):
            raise ValueError(f"相机 {camera_name!r} 的 height 和 width 需要大于或等于 16 的整数")
        height, width = int(height), int(width)
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "fps": fps,
            "shape": (height, width, 3),
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
    if array.ndim != 3 or array.shape[-1] not in (3, 4) or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"camera frame must have shape (H, W, 3/4), got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("camera frame must contain finite numeric pixels")
        if array.max(initial=0) <= 1.0:
            array = np.clip(array * 255.0, 0.0, 255.0)
        else:
            array = np.clip(array, 0.0, 255.0)
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
    import torch

    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(torch.float32).cpu()
    else:
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
    motor = batched_action_to_motor_units(tensor)
    return motor.numpy().astype(np.float32, copy=False)


class OpenSO101LeRobotRecorder:
    """使用 LeRobotDataset 采集双相机 episode。"""

    def __init__(self, repo_id: str, root: str, task_name: str, cameras: Mapping[str, Mapping[str, int]], fps: int):
        self.repo_id = repo_id
        self.root = root
        self.task_name = task_name
        self.cameras = dict(cameras)
        self.features = build_lerobot_features(self.cameras, fps=fps)
        self.fps = int(fps)
        self._dataset = None
        self._recording = False
        self._frames_in_episode = 0
        self._closed = False

    @property
    def recording(self) -> bool:
        return self._recording

    def init_dataset(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from openso101.il.datasets.validation import validate_lerobot_metadata

        if self._closed or self._dataset is not None:
            raise RuntimeError("LeRobot recorder 需要在尚未初始化的状态创建数据集")

        if has_lerobot_metadata(self.root):
            validate_lerobot_metadata(Path(self.root))
            self._dataset = LeRobotDataset(self.repo_id, root=self.root)
            if self._dataset.fps != self.fps:
                raise ValueError("已有 LeRobot 数据集的 FPS 与采集频率不一致")
            for name, expected in self.features.items():
                actual = self._dataset.features[name]
                if any(actual[key] != expected[key] for key in ("dtype", "names")) or tuple(actual["shape"]) != expected["shape"]:
                    raise ValueError(f"已有 LeRobot 数据集的 {name} 与采集配置不一致")
            print(f"[INFO]: 已打开 LeRobot 数据集: {self.root}")
            return

        archived_root = prepare_lerobot_root_for_create(self.root)
        if archived_root is not None:
            print(f"[INFO]: 已保存未完成的数据集: {archived_root}")
        self._dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self.features,
            root=self.root,
            robot_type="so101_follower",
        )
        print(f"[INFO]: 已创建 LeRobot 数据集: {self.root}")

    def start_episode(self) -> None:
        if self._closed or self._recording:
            raise RuntimeError("开始 episode 需要未关闭且没有正在录制的 LeRobot recorder")
        if self._dataset is None:
            self.init_dataset()
        if self._dataset.episode_buffer is None:
            self._dataset.episode_buffer = self._dataset.create_episode_buffer()
        if self._dataset.episode_buffer["size"]:
            raise RuntimeError("LeRobot 开始 episode 时仍有未保存的数据")
        self._recording = True
        self._frames_in_episode = 0
        print("[INFO]: 已开始 LeRobot 录制。")

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
        # LeRobot 保存 action、state 和 RGB；完整仿真状态使用 HDF5 采集。
        del sim_state, qvel, timestamp
        if not self._recording:
            return
        if observation is None:
            observation = qpos
        if observation is None:
            raise ValueError("LeRobot 采集需要 observation 或 qpos")
        if camera_buffers is None:
            raise ValueError("LeRobot 采集需要相机数据")
        ensure_required_cameras(camera_buffers)
        action_motor = _sim_radians_array_to_motor_units(action)
        observation_motor = _sim_radians_array_to_motor_units(observation)
        if any(values.shape != (6,) or not np.isfinite(values).all()
               for values in (action_motor, observation_motor)):
            raise ValueError("LeRobot action 和 observation.state 需要六个有限数值")
        frame = {
            "action": action_motor,
            "observation.state": observation_motor,
            "task": self.task_name,
        }
        for camera_name in REQUIRED_CAMERA_NAMES:
            key = f"observation.images.{camera_name}"
            image = _as_numpy_rgb(camera_buffers[camera_name]).copy()
            if image.shape != self.features[key]["shape"]:
                raise ValueError(f"相机 {camera_name} 的 RGB 尺寸与采集配置不一致")
            frame[key] = image
        self._dataset.add_frame(frame)
        self._frames_in_episode += 1

    def save_episode(self, success: bool = False) -> None:
        if not self._recording:
            return
        if self._frames_in_episode == 0:
            self.cancel_episode()
            return
        self._dataset.save_episode()
        self._recording = False
        print(f"[INFO]: 已保存 LeRobot episode: {self._frames_in_episode} 帧。")

    def cancel_episode(self) -> None:
        if not self._recording:
            return
        self._dataset.clear_episode_buffer()
        self._recording = False
        self._frames_in_episode = 0
        print("[INFO]: 已取消 LeRobot episode。")

    def create_checkpoint(self) -> int:
        if not self._recording:
            raise RuntimeError("创建 checkpoint 需要正在录制的 LeRobot episode")
        if self._dataset.episode_buffer["size"] != self._frames_in_episode:
            raise RuntimeError("LeRobot episode buffer 与 recorder 帧数不一致")
        return self._frames_in_episode

    def restore_checkpoint(self, checkpoint: int) -> None:
        self.create_checkpoint()
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, Integral):
            raise ValueError("LeRobot checkpoint 需要整数帧数")
        if checkpoint < 0 or checkpoint > self._frames_in_episode:
            raise ValueError("LeRobot checkpoint 超出当前 episode 的记录范围")
        buffer = self._dataset.episode_buffer
        self._dataset._wait_image_writer()
        for key in self._dataset.meta.camera_keys:
            for index in range(checkpoint, self._frames_in_episode):
                self._dataset._get_image_file_path(buffer["episode_index"], key, index).unlink()
        for value in buffer.values():
            if isinstance(value, list):
                del value[checkpoint:]
        buffer["size"] = int(checkpoint)
        self._frames_in_episode = int(checkpoint)

    def close(self) -> None:
        if self._closed:
            return
        self.cancel_episode()
        if self._dataset is not None:
            self._dataset.stop_image_writer()
            self._dataset.finalize()
        self._closed = True
