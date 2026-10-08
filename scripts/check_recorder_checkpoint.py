import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from openso101.scenes.models import file_digest
from openso101.teleop.checkpoints import _TeleopCheckpointStore
from openso101.teleop.recorder.hdf5 import OpenSO101HDF5TeleopRecorder, validate_hdf5_episode
from openso101.teleop.simulation import recorded_simulation


def check_case(source, output, checkpoint_frames, written_frames, final_frames):
    cameras = {name: {"height": source[f"observations/images/{name}"].shape[1],
                      "width": source[f"observations/images/{name}"].shape[2]}
               for name in ("wrist_camera", "overhead_camera")}
    recorder = OpenSO101HDF5TeleopRecorder(output, str(source.attrs["task"]), cameras,
                                         int(source.attrs["fps"]), flush_steps=16,
                                         env_id=str(source.attrs["env_id"]),
                                         simulation=recorded_simulation(source.attrs))
    paths = ("action", "observations/qpos", "observations/qvel", "timestamps",
             "observations/images/wrist_camera", "observations/images/overhead_camera",
             *(f"sim/{name}" for name in source["sim"]))
    buffers = {path: np.array(source[path][0], copy=True) for path in paths}

    def add_frame(index):
        for path in paths:
            np.copyto(buffers[path], source[path][index])
        recorder.add_frame(action=buffers["action"], qpos=buffers["observations/qpos"],
                           qvel=buffers["observations/qvel"], timestamp=float(buffers["timestamps"]),
                           camera_buffers={name: buffers[f"observations/images/{name}"] for name in cameras},
                           sim_state={name: buffers[f"sim/{name}"] for name in source["sim"]})

    recorder.start_episode()
    try:
        for index in range(checkpoint_frames):
            add_frame(index)
        checkpoint = recorder.create_checkpoint()
        store = _TeleopCheckpointStore()
        store.capture(recorder)
        for index in range(checkpoint_frames, written_frames):
            add_frame(index)
        with pytest.raises(ValueError, match="整数帧数"):
            recorder.restore_checkpoint(True)
        if store.restore(recorder) is not None:
            raise RuntimeError("记录恢复需要保持独立的 CPU 验证范围")
        if recorder.total_frames != checkpoint_frames:
            raise RuntimeError("HDF5 checkpoint 恢复的帧数不一致")
        for index in range(checkpoint_frames, final_frames):
            add_frame(index)
        # 持续读取后续真实帧，检查录制器保存的数组内容。
        for path in paths:
            np.copyto(buffers[path], source[path][final_frames + 1])
        path = recorder.save_episode(success=False)
    finally:
        recorder._close_file()
    validate_hdf5_episode(path)
    with h5py.File(path, "r") as saved:
        if recorded_simulation(saved.attrs) != recorded_simulation(source.attrs):
            raise ValueError("HDF5 checkpoint 后的仿真参数与实际来源不一致")
        for key in paths:
            if saved[key].shape[0] != final_frames or not np.array_equal(saved[key][:], source[key][:final_frames]):
                raise RuntimeError(f"HDF5 checkpoint 后的实际数据不一致: {key}")
        if saved["checkpoints/frame_index"][:].tolist() != [checkpoint_frames - 1] or saved.attrs["success"]:
            raise RuntimeError("HDF5 checkpoint 的帧索引或任务标记不一致")
    with pytest.raises(RuntimeError, match="正在录制"):
        recorder.restore_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match="正在录制"):
        recorder.create_checkpoint()
    return {"checkpoint_frames": checkpoint_frames, "written_frames_before_restore": written_frames,
            "saved_frames": final_frames, "actual_fields": len(paths), "all_arrays_equal": True,
            "saved_episode": str(path), "sha256": file_digest(path), "task_success_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("录制器 checkpoint 检查需要禁止 CUDA")
    validate_hdf5_episode(args.episode)
    digest = file_digest(args.episode)
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.episode, "r") as source:
        cases = {"disk": check_case(source, args.output / "disk", 8, 24, 16),
                 "buffer": check_case(source, args.output / "buffer", 20, 25, 32)}
        cameras = {name: {"height": source[f"observations/images/{name}"].shape[1],
                          "width": source[f"observations/images/{name}"].shape[2]}
                   for name in ("wrist_camera", "overhead_camera")}
        for value in (0, True, 50.5, float("nan")):
            with pytest.raises(ValueError, match="fps 必须为正整数"):
                OpenSO101HDF5TeleopRecorder(args.output / "invalid_fps", "PickPlace", cameras, value)
    if (args.output / "invalid_fps").exists() or file_digest(args.episode) != digest:
        raise RuntimeError("HDF5 输入检查创建了目录或更改了来源文件")
    report = {"status": "actual_hdf5_checkpoint_verified", "cases": cases,
              "source_episode": str(args.episode), "source_sha256": digest, "source_unchanged": True,
              "input_arrays_preserved": True, "invalid_fps_rejected": True,
              "inactive_checkpoint_rejected": True, "gpu_tests_started": False,
              "native_restore_verified": False, "task_success_verified": False,
              "checkpoint_store_source_sha256": file_digest(Path("src/openso101/teleop/checkpoints.py")),
              "recorded_simulation_preserved": True,
              "recorder_source_sha256": file_digest(Path("src/openso101/teleop/recorder/hdf5.py")),
              "validation_source_sha256": file_digest(Path(__file__))}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "input_arrays_preserved", "gpu_tests_started")}))


if __name__ == "__main__":
    main()
