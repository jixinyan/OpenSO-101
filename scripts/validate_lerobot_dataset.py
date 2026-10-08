import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openso101.il.datasets.export import _push_hdf5_radians_to_motor_units, _push_validate_local_dataset
from openso101.il.observations import camera_to_policy
from openso101.rl.config import digest
from openso101.teleop.recorder.hdf5 import validate_hdf5_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--require-success", action="store_true")
parser.add_argument("--maximum-encoding-mae", type=float, default=0.05)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
if not 0 < args.maximum_encoding_mae < 1:
    raise ValueError("maximum_encoding_mae 需要位于零与一之间")
sources = {path.name: path for path in validate_hdf5_dataset(args.source)}
metadata = json.loads((args.dataset / "meta/openso101_export.json").read_text())
_push_validate_local_dataset(args.dataset, input_format="lerobot")
dataset = LeRobotDataset(metadata["repo_id"], root=args.dataset, download_videos=False)
expected_count = sum(item["exported_frames"] for item in metadata["episodes"])
if len(dataset) != expected_count or dataset.meta.total_episodes != len(metadata["episodes"]):
    raise ValueError("LeRobot 全部帧数或 episode 数量不一致")
frames, action_error, state_error, time_error = 0, 0., 0., 0.
camera_checks = {name: {"frames": 0, "minimum_pixel_std": None, "maximum_encoding_mae": 0.}
                 for name in ("wrist_camera", "overhead_camera")}
for item in metadata["episodes"]:
    source = sources[item["source_episode"]]
    if digest(source) != item["source_sha256"]:
        raise ValueError("LeRobot 来源 SHA256 不一致")
    with h5py.File(source) as recording:
        if bool(recording.attrs["success"]) != item["success"]:
            raise ValueError("LeRobot 成功记录与来源不一致")
        if args.require_success and not item["success"]:
            raise ValueError("成功数据集包含未成功的 episode")
        for source_frame in range(metadata["skip_leading_frames"], item["source_frames"]):
            frame = dataset[frames]
            local_frame = source_frame - metadata["skip_leading_frames"]
            if int(frame["episode_index"]) != item["episode_index"] or int(frame["frame_index"]) != local_frame:
                raise ValueError("LeRobot 帧索引与来源不一致")
            expected_action = _push_hdf5_radians_to_motor_units(recording["action"][source_frame])
            expected_state = _push_hdf5_radians_to_motor_units(recording["observations/qpos"][source_frame])
            action_error = max(action_error, float(np.abs(frame["action"].numpy() - expected_action).max()))
            state_error = max(state_error, float(np.abs(frame["observation.state"].numpy() - expected_state).max()))
            time_error = max(time_error, abs(float(frame["timestamp"]) - local_frame / metadata["fps"]))
            if action_error > 1e-4 or state_error > 1e-4 or time_error > 1e-6:
                raise ValueError("LeRobot 动作、状态或时间与来源不一致")
            for name, check in camera_checks.items():
                rgb = frame[f"observation.images.{name}"]
                pixels = torch.from_numpy(recording[f"observations/images/{name}"][source_frame])
                expected = camera_to_policy(pixels.unsqueeze(0))[0]
                if not torch.equal(expected, pixels.permute(2, 0, 1).float() / 255.):
                    raise ValueError("IL observation 相机转换与来源数值不一致")
                if rgb.shape != expected.shape or not torch.isfinite(rgb).all() or float(rgb.min()) < 0 or float(rgb.max()) > 1:
                    raise ValueError(f"LeRobot 相机数据格式错误: {name}")
                converted = camera_to_policy(rgb.permute(1, 2, 0).unsqueeze(0))[0]
                if not torch.equal(converted, rgb):
                    raise ValueError(f"IL observation 浮点 RGB 与实际视频读取不一致: {name}")
                pixel_std = float(rgb.std())
                if pixel_std <= 0:
                    raise ValueError(f"LeRobot 相机数据没有像素变化: {name}")
                previous = check["minimum_pixel_std"]
                check["minimum_pixel_std"] = pixel_std if previous is None else min(previous, pixel_std)
                check["maximum_encoding_mae"] = max(check["maximum_encoding_mae"], float((rgb - expected).abs().mean()))
                if check["maximum_encoding_mae"] > args.maximum_encoding_mae:
                    raise ValueError(f"LeRobot 视频编码误差超过要求: {name}")
                check["frames"] += 1
            frames += 1
result = {"status": "all_lerobot_frames_verified", "frames": frames,
          "episodes": len(metadata["episodes"]), "source_task_successes": sum(item["success"] for item in metadata["episodes"]),
          "maximum_action_error_motor_units": action_error, "maximum_state_error_motor_units": state_error,
          "maximum_timestamp_error_seconds": time_error, "camera_checks": camera_checks,
          "maximum_encoding_mae_allowed": args.maximum_encoding_mae,
          "policy_observation_uint8_and_float_verified": True,
          "export_manifest_sha256": digest(args.dataset / "meta/openso101_export.json"),
          "validator_sha256": digest(Path(__file__)), "rl_policy_success_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2), flush=True)
