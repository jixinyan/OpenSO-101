import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from openso101.scenes.models import file_digest
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode
from openso101.teleop.sim_state import _replay_optional_frame, _replay_to_tensor_like, _tensor_to_numpy
from openso101.teleop.timing import control_rate_fps
from openso101.teleop.state_records import replay_root_state, validate_replay_task_state


parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("已记录状态的 CPU 检查需要禁止 CUDA")
if args.output.exists():
    raise FileExistsError(args.output)
validate_hdf5_episode(args.episode)
source_hash = file_digest(args.episode)
with h5py.File(args.episode, "r") as recording:
    frames = len(recording["action"])
    fps, physics_dt = int(recording.attrs["fps"]), float(recording.attrs["physics_dt"])
    decimation = round(1 / (fps * physics_dt))
    if control_rate_fps(physics_dt, decimation) != fps:
        raise ValueError("实际录制 FPS 与 physics_dt 不一致")
    fields = [f"sim/{name}" for name in recording["sim"]]
    if not fields:
        raise ValueError("实际录制需要包含场景状态")
    errors = {field: 0.0 for field in fields}
    validate_replay_task_state(recording, str(recording.attrs["env_id"]))
    root_state_errors, root_state_transforms = [], 0
    for frame_index in range(frames):
        original = recording["sim/object_root_state"][frame_index]
        origin = recording["sim/environment_origin"][frame_index]
        for destination in recording["sim/cohort_environment_origins"][frame_index]:
            restored = replay_root_state(original, origin, destination)
            recovered = replay_root_state(restored, destination, origin)
            error = max(float(np.abs(recovered - original).max()),
                        float(np.abs((restored[:3] - destination) - (original[:3] - origin)).max()))
            if error > 0.000001 or not np.array_equal(restored[3:], original[3:]):
                raise ValueError("实际物体的回放坐标、quaternion 或速度转换不一致")
            root_state_errors.append(error)
            root_state_transforms += 1
    for field in fields:
        for frame_index in range(frames):
            expected = np.asarray(recording[field][frame_index])
            if not np.isfinite(expected).all():
                raise ValueError(f"已记录状态包含无效数值: {field}")
            reference = torch.as_tensor(expected.copy())
            value = _replay_to_tensor_like(_replay_optional_frame(recording, field, frame_index), reference)
            actual = _tensor_to_numpy(value)
            if value.device.type != "cpu" or not np.array_equal(actual, expected):
                raise ValueError(f"回放状态的 CPU 转换与来源不一致: {field}")
            errors[field] = max(errors[field], float(np.abs(actual.astype(float) - expected.astype(float)).max()))
if file_digest(args.episode) != source_hash:
    raise ValueError("检查过程中来源文件发生改变")
task_rejections = []
for task in ("OpenSO101-Lift-v0", "OpenSO101-Stack-v0"):
    process = subprocess.run([sys.executable, "-m", "openso101.cli.main", "il", "replay",
                              "--episode", str(args.episode), "--task", task, "--headless"],
                             capture_output=True, text=True)
    if process.returncode == 0 or "回放任务需要与记录的 env_id 一致" not in process.stderr or "AppLauncher" in process.stderr:
        raise RuntimeError("实际回放入口需要在启动 Isaac 前检查来源任务")
    task_rejections.append({"task": task, "exit_code": process.returncode})
capture_output = args.output.with_suffix(".capture")
process = subprocess.run([sys.executable, "scripts/capture_agent_scene.py", "unused_scene",
                          str(args.episode), str(capture_output), "--steps", "0"],
                         capture_output=True, text=True)
if process.returncode == 0 or "steps 必须为正数" not in process.stderr or capture_output.exists():
    raise RuntimeError("场景录制需要在启动 Isaac 前拒绝零步骤")
result = {"status": "recorded_state_cpu_conversion_verified", "frames": frames, "fps": fps,
          "physics_dt": physics_dt, "derived_decimation": decimation, "field_errors": errors,
          "episode_sha256": source_hash, "source_sha256": file_digest(Path(__file__)),
          "sim_state_sha256": file_digest(Path("src/openso101/teleop/sim_state.py")),
          "timing_sha256": file_digest(Path("src/openso101/teleop/timing.py")),
          "root_state_transforms": root_state_transforms,
          "root_state_maximum_error": max(root_state_errors),
          "root_state_source_components_preserved": True,
          "replay_task_preflight_verified": True, "rejected_tasks": task_rejections,
          "state_records_sha256": file_digest(Path("src/openso101/teleop/state_records.py")),
          "gpu_tests_started": False, "native_restore_verified": False,
          "capture_count_preflight_verified": True,
          "capture_source_sha256": file_digest(Path("scripts/capture_agent_scene.py"))}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))
