from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openso101.scenes.models import file_digest


_DEFAULT_SKIP_LEADING_FRAMES = 5
_DEFAULT_MIN_EPISODE_FRAMES = 10


def _push_hdf5_radians_to_motor_units(values):
    import numpy as np
    import torch

    from openso101.teleop.so101_mapping import batched_action_to_motor_units

    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32))
    return batched_action_to_motor_units(tensor).numpy().astype(np.float32, copy=False)


def _push_relative_missing_lerobot_paths(root: Path) -> list[str]:
    required = (Path("meta/info.json"), Path("meta/tasks.parquet"), Path("meta/stats.json"))
    return [str(relative) for relative in required if not (root / relative).is_file()]


def _push_lerobot_episode_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("**/*.parquet"))


def _push_detect_input_format(root: Path) -> str:
    if (root / "episodes").is_dir() and list((root / "episodes").glob("episode_*.hdf5")):
        return "hdf5"
    if (root / "meta").is_dir() or (root / "data").is_dir():
        return "lerobot"
    raise ValueError(f"无法识别数据集格式: {root}")


def _push_camera_metadata_from_hdf5_episode(episode_file: Path) -> dict[str, dict[str, int]]:
    import h5py
    import numpy as np

    with h5py.File(episode_file, "r") as h5:
        cameras = {}
        for name in ("wrist_camera", "overhead_camera"):
            pixels = h5[f"observations/images/{name}"]
            height, width = pixels.shape[1:3]
            if height < 16 or width < 16 or height % 2 or width % 2:
                raise ValueError(f"{episode_file} 的 {name} 视频尺寸需要至少 16 × 16，并且能够被 2 整除")
            if pixels.dtype != np.uint8:
                raise ValueError(f"{episode_file} 的 {name} 需要 uint8 RGB 数据")
            cameras[name] = {"height": int(height), "width": int(width)}
        return cameras


def _push_features_from_hdf5_episode(episode_file: Path, fps: int) -> dict[str, dict]:
    from openso101.teleop.so101_mapping import LEROBOT_SO101_ACTION_NAMES

    joint_feature = {"dtype": "float32", "fps": fps, "shape": (6,),
                     "names": list(LEROBOT_SO101_ACTION_NAMES)}
    features = {"observation.state": dict(joint_feature), "action": dict(joint_feature)}
    for name, camera in _push_camera_metadata_from_hdf5_episode(episode_file).items():
        features[f"observation.images.{name}"] = {
            "dtype": "video", "fps": fps, "shape": (camera["height"], camera["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _push_archive_existing_export(root: Path) -> Path:
    archive = root.with_name(f"{root.name}.previous-export")
    suffix = 1
    while archive.exists():
        archive = root.with_name(f"{root.name}.previous-export-{suffix}")
        suffix += 1
    root.rename(archive)
    return archive


def _push_validate_local_dataset(root: Path, input_format: str = "auto") -> list[Path]:
    from openso101.teleop.recorder.hdf5 import validate_hdf5_dataset

    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    resolved = _push_detect_input_format(root) if input_format == "auto" else input_format
    if resolved == "hdf5":
        episodes = validate_hdf5_dataset(root)
        for episode in episodes:
            _push_camera_metadata_from_hdf5_episode(episode)
        return episodes
    if resolved != "lerobot":
        raise ValueError(f"无法识别数据集格式: {resolved}")
    missing = _push_relative_missing_lerobot_paths(root)
    if missing:
        raise ValueError(f"LeRobot metadata 缺少文件: {', '.join(missing)}")
    episodes = _push_lerobot_episode_files(root)
    if not episodes:
        raise ValueError(f"LeRobot 数据集没有 episode Parquet: {root / 'data'}")
    from openso101.il.datasets.validation import validate_lerobot_metadata

    return validate_lerobot_metadata(root)


def _push_convert_hdf5_to_lerobot(
    hdf5_root: Path,
    lerobot_root: Path,
    repo_id: str,
    overwrite_export: bool = False,
    skip_leading_frames: int = _DEFAULT_SKIP_LEADING_FRAMES,
    min_episode_frames: int = _DEFAULT_MIN_EPISODE_FRAMES,
    async_flush: bool = True,
    include_failures: bool = False,
) -> Path:
    import h5py
    import numpy as np

    from openso101.scenes.recording import resolve_recording_scene, store_recording_scene
    from openso101.teleop.recorder.hdf5 import validate_hdf5_dataset
    from openso101.teleop.simulation import recorded_simulation

    if skip_leading_frames < 0 or min_episode_frames < 1:
        raise ValueError("skip_leading_frames 需要大于或等于零，min_episode_frames 需要大于零")
    hdf5_root, lerobot_root = hdf5_root.resolve(), lerobot_root.resolve()
    if hdf5_root == lerobot_root or lerobot_root in hdf5_root.parents:
        raise ValueError("导出目录需要独立于来源目录及其父目录")
    episode_files = validate_hdf5_dataset(hdf5_root)
    if any(episode.resolve().is_relative_to(lerobot_root) for episode in episode_files):
        raise ValueError("导出目录不能包含实际来源 episode")
    skipped_failed, skipped_short, records = [], [], []
    fps, cameras = None, None
    # 创建输出目录之前检查全部来源与过滤结果。
    for episode in episode_files:
        with h5py.File(episode, "r") as h5:
            success = bool(h5.attrs.get("success", False))
            if not include_failures and not success:
                skipped_failed.append(episode.name)
                continue
            count = int(h5["action"].shape[0])
            usable = count - skip_leading_frames
            if usable < min_episode_frames:
                skipped_short.append(episode.name)
                continue
            source_fps = float(h5.attrs["fps"])
            if not source_fps.is_integer():
                raise ValueError(f"LeRobot 导出要求整数 FPS: {episode}")
            source_cameras = _push_camera_metadata_from_hdf5_episode(episode)
            if fps is not None and (fps != int(source_fps) or cameras != source_cameras):
                raise ValueError(f"episode 的 FPS 或双相机尺寸不一致: {episode}")
            fps, cameras = int(source_fps), source_cameras
            scene = resolve_recording_scene(episode, h5.attrs) if "scene_sha256" in h5.attrs else None
            simulation = recorded_simulation(h5.attrs)
            if scene is not None and scene.resolve().is_relative_to(lerobot_root):
                raise ValueError("导出目录不能包含来源场景")
            records.append({"episode_index": len(records), "source_episode": episode.name,
                            "source_sha256": file_digest(episode), "source_frames": count,
                            "exported_frames": usable, "success": success,
                            "task": str(h5.attrs.get("task", "OpenSO-101 teleoperation")),
                            "env_id": str(h5.attrs.get("env_id", "")), "scene": scene,
                            "simulation": simulation.model_dump(mode="json") if simulation is not None else None})
    if not records:
        raise ValueError("全部 episode 均未满足成功标记与帧数要求")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = _push_features_from_hdf5_episode(hdf5_root / "episodes" / records[0]["source_episode"], fps)
    if lerobot_root.exists():
        if not overwrite_export:
            raise FileExistsError(f"导出目录已存在: {lerobot_root}")
        archive = _push_archive_existing_export(lerobot_root)
        print(f"[INFO]: 已保存原有导出目录: {archive}")
    dataset = LeRobotDataset.create(repo_id, fps=fps, features=features,
                                   root=lerobot_root, robot_type="so101_follower")
    scene_records = []
    pending = None
    # 每次仅允许一个 save_episode，关闭时等待实际编码完成并传播异常。
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="lerobot-flush") as executor:
        for record in records:
            episode = hdf5_root / "episodes" / record["source_episode"]
            if record["scene"] is not None:
                metadata = store_recording_scene(record["scene"], lerobot_root)
                scene_records.append({"episode_index": record["episode_index"],
                                      "source_episode": episode.name, "success": record["success"], **metadata})
            with h5py.File(episode, "r") as h5:
                actions = _push_hdf5_radians_to_motor_units(h5["action"][skip_leading_frames:])
                states = _push_hdf5_radians_to_motor_units(h5["observations/qpos"][skip_leading_frames:])
                if pending is not None:
                    pending.result()
                for local_index, source_index in enumerate(range(skip_leading_frames, record["source_frames"])):
                    frame = {"action": actions[local_index], "observation.state": states[local_index],
                             "task": record["task"]}
                    for name in cameras:
                        frame[f"observation.images.{name}"] = np.asarray(h5[f"observations/images/{name}"][source_index])
                    dataset.add_frame(frame)
                if async_flush:
                    pending = executor.submit(dataset.save_episode)
                else:
                    dataset.save_episode()
                if file_digest(episode) != record["source_sha256"]:
                    raise ValueError(f"导出过程中来源文件发生改变: {episode}")
        if pending is not None:
            pending.result()
    dataset.finalize()
    if scene_records:
        (lerobot_root / "meta/scenes.json").write_text(json.dumps(scene_records, indent=2), encoding="utf-8")
    exported_records = [{key: value for key, value in record.items() if key != "scene"} for record in records]
    (lerobot_root / "meta/openso101_export.json").write_text(json.dumps({
        "schema_version": 1, "repo_id": repo_id, "fps": fps,
        "skip_leading_frames": skip_leading_frames, "min_episode_frames": min_episode_frames,
        "include_failures": include_failures, "async_flush": async_flush,
        "episodes": exported_records, "skipped_failed": skipped_failed, "skipped_short": skipped_short,
        "exporter_sha256": file_digest(Path(__file__)),
    }, indent=2), encoding="utf-8")
    print(f"[INFO]: 已导出 {len(records)} 个 episode，共 {sum(item['exported_frames'] for item in records)} 帧")
    return lerobot_root
