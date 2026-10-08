from pathlib import Path


def validate_lerobot_metadata(root: Path) -> list[Path]:
    import numpy as np
    import pyarrow.parquet as parquet
    from packaging.version import Version
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    from lerobot.datasets.utils import load_episodes, load_info, load_stats, load_tasks

    from openso101.teleop.so101_mapping import LEROBOT_SO101_ACTION_NAMES

    info = load_info(root)
    if Version(info["codebase_version"]).major != Version(CODEBASE_VERSION).major:
        raise ValueError("LeRobot 数据版本与当前库的 major version 不一致")
    for name in ("fps", "total_frames", "total_episodes", "total_tasks"):
        value = info[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"LeRobot {name} 必须为正整数")
    features = info["features"]
    for name in ("action", "observation.state"):
        feature = features[name]
        if feature["dtype"] != "float32" or tuple(feature["shape"]) != (6,):
            raise ValueError(f"LeRobot {name} 需要六个 float32 关节值")
        if tuple(feature["names"]) != LEROBOT_SO101_ACTION_NAMES:
            raise ValueError(f"LeRobot {name} 关节顺序不一致")
    cameras = ("observation.images.wrist_camera", "observation.images.overhead_camera")
    for name in cameras:
        feature = features[name]
        shape = feature["shape"]
        if feature["dtype"] != "video" or len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 16:
            raise ValueError(f"LeRobot {name} 需要 RGB 视频")
        if feature["info"]["video.fps"] != info["fps"]:
            raise ValueError(f"LeRobot {name} 视频 FPS 不一致")
    statistics = load_stats(root)
    if statistics is None:
        raise ValueError("LeRobot 数据集缺少统计量")
    for name in ("action", "observation.state", *cameras):
        for statistic in ("min", "max", "mean", "std", "count"):
            values = np.asarray(statistics[name][statistic])
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"LeRobot {name}/{statistic} 需要有限数值")
            if statistic == "count" and (values <= 0).any():
                raise ValueError(f"LeRobot {name}/count 必须大于零")
            if statistic == "std" and (values < 0).any():
                raise ValueError(f"LeRobot {name}/std 必须大于或等于零")
    tasks = load_tasks(root)
    episodes = load_episodes(root)
    if len(tasks) != info["total_tasks"] or len(episodes) != info["total_episodes"]:
        raise ValueError("LeRobot task 或 episode 数量与 metadata 不一致")
    if "task_index" not in tasks or sorted(tasks["task_index"].tolist()) != list(range(len(tasks))):
        raise ValueError("LeRobot task_index 必须连续")
    files, next_index = set(), 0
    for index, episode in enumerate(episodes):
        if episode["episode_index"] != index or episode["dataset_from_index"] != next_index:
            raise ValueError("LeRobot episode 与数据范围必须连续")
        count = episode["dataset_to_index"] - next_index
        if count < 1 or count != episode["length"]:
            raise ValueError("LeRobot episode 帧数与数据范围不一致")
        next_index += count
        files.add(_dataset_file(root, info["data_path"].format(
            chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])))
        for name in cameras:
            _dataset_file(root, info["video_path"].format(video_key=name,
                chunk_index=episode[f"videos/{name}/chunk_index"], file_index=episode[f"videos/{name}/file_index"]))
    columns = {"action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    frames = 0
    for path in sorted(files):
        metadata = parquet.read_metadata(path)
        if metadata.num_rows < 1 or not columns.issubset(parquet.read_schema(path).names):
            raise ValueError(f"LeRobot 数据 Parquet 的列或帧数不完整: {path}")
        frames += metadata.num_rows
    if frames != next_index or frames != info["total_frames"]:
        raise ValueError("LeRobot 实际数据帧数、episode 范围与 total_frames 不一致")
    return sorted(files)


def _dataset_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"LeRobot 文件需要位于数据集目录: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"LeRobot 文件不存在或内容为空: {path}")
    return path
