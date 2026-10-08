import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openso101.il.datasets.validation import validate_lerobot_metadata
from openso101.scenes.models import file_digest
from openso101.teleop.checkpoints import _TeleopCheckpointStore
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode
from openso101.teleop.recorder.lerobot import OpenSO101LeRobotRecorder, _sim_radians_array_to_motor_units


def validate_frames(root, repo_id, source, segments):
    validate_lerobot_metadata(root)
    dataset = LeRobotDataset(repo_id, root=root, download_videos=False)
    expected_count = sum(end - start for start, end in segments)
    if len(dataset) != expected_count or dataset.meta.total_episodes != len(segments):
        raise RuntimeError("LeRobot 保存的实际帧数或 episode 数量不一致")
    action_error, state_error, time_error = 0.0, 0.0, 0.0
    camera_errors = {name: 0.0 for name in ("wrist_camera", "overhead_camera")}
    global_index = 0
    for episode_index, (start, end) in enumerate(segments):
        for local_index, source_index in enumerate(range(start, end)):
            frame = dataset[global_index]
            if int(frame["episode_index"]) != episode_index or int(frame["frame_index"]) != local_index:
                raise RuntimeError("LeRobot 采集的实际 episode 或帧索引不一致")
            expected_action = _sim_radians_array_to_motor_units(source["action"][source_index])
            expected_state = _sim_radians_array_to_motor_units(source["observations/qpos"][source_index])
            action_error = max(action_error, float(np.abs(frame["action"].numpy() - expected_action).max()))
            state_error = max(state_error, float(np.abs(frame["observation.state"].numpy() - expected_state).max()))
            time_error = max(time_error, abs(float(frame["timestamp"]) - local_index / dataset.fps))
            for name in camera_errors:
                image = frame[f"observation.images.{name}"]
                expected = torch.from_numpy(source[f"observations/images/{name}"][source_index]).permute(2, 0, 1).float() / 255
                if image.shape != expected.shape or not torch.isfinite(image).all() or float(image.std()) <= 0:
                    raise RuntimeError(f"LeRobot 采集的视频帧内容不完整: {name}")
                camera_errors[name] = max(camera_errors[name], float((image - expected).abs().mean()))
            global_index += 1
    if action_error > 0.0001 or state_error > 0.0001 or time_error > 0.000001 or max(camera_errors.values()) > 0.05:
        raise RuntimeError("LeRobot 采集的动作、状态、时间或视频与实际来源不一致")
    dataset.finalize()
    return {"frames": global_index, "episodes": len(segments),
            "maximum_action_error_motor_units": action_error, "maximum_state_error_motor_units": state_error,
            "maximum_timestamp_error_seconds": time_error, "camera_maximum_encoding_mae": camera_errors}


def check_case(source, root, asynchronous):
    cameras = {name: {"height": source[f"observations/images/{name}"].shape[1],
                      "width": source[f"observations/images/{name}"].shape[2]}
               for name in ("wrist_camera", "overhead_camera")}
    repo_id = f"local/recorder_{'async' if asynchronous else 'sync'}"
    fps = int(source.attrs["fps"])
    recorder = OpenSO101LeRobotRecorder(repo_id, str(root), str(source.attrs["task"]), cameras, fps)
    recorder.init_dataset()
    if asynchronous:
        recorder._dataset.start_image_writer(num_threads=4)
    buffers = {name: np.array(source[f"observations/images/{name}"][0], copy=True) for name in cameras}

    def add_frame(index):
        for name, image in buffers.items():
            np.copyto(image, source[f"observations/images/{name}"][index])
        recorder.add_frame(action=source["action"][index], qpos=source["observations/qpos"][index],
                           camera_buffers=buffers, timestamp=float(source["timestamps"][index]))
        # 相机采集会重复使用数组，后续读取需要保持已提交帧的内容。
        for name, image in buffers.items():
            np.copyto(image, source[f"observations/images/{name}"][-1])

    try:
        with pytest.raises(RuntimeError, match="正在录制"):
            recorder.create_checkpoint()
        for start, end, checkpoint_size in ((0, 164, 64), (164, 328, 32)):
            recorder.start_episode()
            with pytest.raises(RuntimeError, match="没有正在录制"):
                recorder.start_episode()
            for index in range(start, start + checkpoint_size):
                add_frame(index)
            checkpoint = recorder.create_checkpoint()
            store = _TeleopCheckpointStore()
            store.capture(recorder)
            for index in range(start + checkpoint_size, start + checkpoint_size + 32):
                add_frame(index)
            episode_index = recorder._dataset.episode_buffer["episode_index"]
            retained = [recorder._dataset._get_image_file_path(episode_index, key, checkpoint - 1)
                        for key in recorder._dataset.meta.camera_keys]
            discarded = [recorder._dataset._get_image_file_path(episode_index, key, index)
                         for key in recorder._dataset.meta.camera_keys
                         for index in range(checkpoint, checkpoint + 32)]
            for invalid in (True, 1.5, -1, checkpoint + 1000):
                with pytest.raises(ValueError):
                    recorder.restore_checkpoint(invalid)
            if store.restore(recorder) is not None:
                raise RuntimeError("记录恢复需要保持独立的 CPU 验证范围")
            if any(path.exists() for path in discarded) or not all(path.is_file() for path in retained):
                raise RuntimeError("LeRobot checkpoint 的相机文件裁剪不一致")
            if recorder.create_checkpoint() != checkpoint:
                raise RuntimeError("LeRobot checkpoint 恢复后的帧数不一致")
            for index in range(start + checkpoint_size, end):
                add_frame(index)
            recorder.save_episode()
        recorder.start_episode()
        add_frame(0)
        pending = [recorder._dataset._get_image_file_path(2, key, 0)
                   for key in recorder._dataset.meta.camera_keys]
        recorder.cancel_episode()
        if any(path.exists() for path in pending) or recorder._dataset.episode_buffer["size"]:
            raise RuntimeError("LeRobot 取消 episode 后仍有待保存的数据")
    finally:
        recorder.close()
    if recorder._dataset.image_writer is not None or recorder._dataset.writer is not None or recorder._dataset.meta.writer is not None:
        raise RuntimeError("LeRobot recorder 关闭后仍有编码或 Parquet writer")
    with pytest.raises(RuntimeError, match="未关闭"):
        recorder.start_episode()
    first = validate_frames(root, repo_id, source, [(0, 164), (164, 328)])
    saved_files = {path: file_digest(path) for parent in (root / "data", root / "videos")
                   for path in parent.rglob("*") if path.is_file()}
    reopened = OpenSO101LeRobotRecorder(repo_id, str(root), str(source.attrs["task"]), cameras, fps)
    try:
        reopened.start_episode()
        for index in range(8):
            reopened.add_frame(action=source["action"][index], qpos=source["observations/qpos"][index],
                               camera_buffers={name: source[f"observations/images/{name}"][index] for name in cameras})
        reopened.save_episode()
    finally:
        reopened.close()
    if any(file_digest(path) != digest for path, digest in saved_files.items()):
        raise RuntimeError("重新打开 LeRobot 采集更改了已有数据和视频")
    result = validate_frames(root, repo_id, source, [(0, 164), (164, 328), (0, 8)])
    mismatch = OpenSO101LeRobotRecorder(repo_id, str(root), str(source.attrs["task"]), cameras, fps + 1)
    try:
        with pytest.raises(ValueError, match="FPS"):
            mismatch.init_dataset()
    finally:
        mismatch.close()
    result.update({"original_frames_checked": first["frames"], "checkpoint_camera_suffix_removed": True,
                   "cancelled_episode_removed": True, "reopened_dataset_preserved": True,
                   "input_camera_arrays_preserved": True, "writers_closed": True,
                   "files": {str(path.relative_to(root)): file_digest(path) for path in root.rglob("*") if path.is_file()}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("LeRobot 录制检查需要禁止 CUDA")
    validate_hdf5_episode(args.episode)
    source_digest = file_digest(args.episode)
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.episode, "r") as source:
        if len(source["action"]) != 328:
            raise ValueError("当前验证需要实际 328 帧来源 episode")
        cases = {"sync": check_case(source, args.output / "sync", False),
                 "async": check_case(source, args.output / "async", True)}
        cameras = {name: {"height": source[f"observations/images/{name}"].shape[1],
                          "width": source[f"observations/images/{name}"].shape[2]}
                   for name in ("wrist_camera", "overhead_camera")}
        for value in (0, True, 50.5, float("nan")):
            with pytest.raises(ValueError, match="fps 必须为正整数"):
                OpenSO101LeRobotRecorder("local/invalid", str(args.output / "invalid"), "PickPlace", cameras, value)
    if file_digest(args.episode) != source_digest or (args.output / "invalid").exists():
        raise RuntimeError("LeRobot 检查更改了来源或创建了无效目录")
    report = {"status": "actual_lerobot_recorder_verified", "cases": cases,
              "source_episode": str(args.episode), "source_sha256": source_digest,
              "source_unchanged": True, "gpu_tests_started": False, "native_restore_verified": False,
              "task_success_verified": False,
              "checkpoint_store_source_sha256": file_digest(Path("src/openso101/teleop/checkpoints.py")),
              "recorder_source_sha256": file_digest(Path("src/openso101/teleop/recorder/lerobot.py")),
              "validation_source_sha256": file_digest(Path(__file__))}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "source_unchanged", "gpu_tests_started")}))


if __name__ == "__main__":
    main()
