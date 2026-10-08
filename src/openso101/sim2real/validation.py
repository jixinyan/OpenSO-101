import json
from pathlib import Path

import h5py
import numpy as np
import torch

from openso101.rl.config import digest
from openso101.rl.student import RLStudentPolicy
from openso101.robots.so101.constants import SO101_SIM_JOINT_NAMES
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode
from openso101.teleop.so101_mapping import batched_action_to_motor_units

from .deploy import _clamp_motor_units


def validate(args):
    if args.batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    folder = Path(args.policy_path).resolve()
    episode = Path(args.episode).resolve()
    validate_hdf5_episode(episode)
    policy = RLStudentPolicy(folder, args.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    with h5py.File(episode, "r") as recording:
        metadata = policy.metadata
        if not np.isclose(1 / recording.attrs["fps"], metadata["control_dt"], rtol=0, atol=1e-6):
            raise ValueError("录制控制周期与 student 不一致")
        if recording.attrs["env_id"] != metadata["task_id"]:
            raise ValueError("录制任务与 student 任务不一致")
        if recording.attrs.get("scene_sha256") != metadata.get("scene_sha256"):
            raise ValueError("录制场景与 student 场景不一致")
        if tuple(recording.attrs["sim_joint_names"]) != SO101_SIM_JOINT_NAMES:
            raise ValueError("录制关节顺序不一致")
        frames = len(recording["action"])
        for start in range(0, frames, args.batch_size):
            end = min(start + args.batch_size, frames)
            qpos = torch.as_tensor(recording["observations/qpos"][start:end], device=args.device, dtype=torch.float32)
            observation = {"observation.state": batched_action_to_motor_units(qpos)}
            if policy.model.goal_dim:
                observation["observation.goal"] = torch.as_tensor(recording["sim/task_goal_root"][start:end],
                                                                    device=args.device, dtype=torch.float32)
            for name in ("wrist_camera", "overhead_camera"):
                images = torch.as_tensor(recording[f"observations/images/{name}"][start:end], device=args.device)
                observation[f"observation.images.{name}"] = images.permute(0, 3, 1, 2).float() / 255.
            with torch.inference_mode():
                actions = policy.decode_actions(policy.select_action(policy.preprocess(observation))).cpu().numpy()
            commands.extend(_clamp_motor_units(action) for action in actions)
    commands = np.asarray(commands)
    if commands.shape != (frames, 6) or not np.isfinite(commands).all():
        raise RuntimeError("student 推理结果形状或数值无效")
    np.save(output / "motor_commands.npy", commands)
    report = {
        "status": "student_recorded_observation_inference_verified", "frames": frames,
        "task": metadata["task_id"], "scene_sha256": metadata.get("scene_sha256"),
        "student_sha256": metadata["files"]["student.pt"], "episode_sha256": digest(episode),
        "control_dt": metadata["control_dt"], "required_fps": 1 / metadata["control_dt"],
        "motor_minimum": commands.min(axis=0).tolist(), "motor_maximum": commands.max(axis=0).tolist(),
        "commands_sha256": digest(output / "motor_commands.npy"), "hardware_run_verified": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0
