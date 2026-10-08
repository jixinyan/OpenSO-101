import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from openso101.rl.config import CheckpointMeta, digest
from openso101.rl.gpu_scope import configure_visible_gpu
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--episode", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--arm-target-speed-limit", type=float)
args = parser.parse_args()
if args.arm_target_speed_limit is not None and not 0 < args.arm_target_speed_limit <= 1.5:
    raise ValueError("arm 目标速度需要大于零且不超过实际速度限制 1.5 rad/s")
configure_visible_gpu()
if args.output.exists():
    raise FileExistsError(args.output)
meta = CheckpointMeta.read(args.checkpoint)
validate_hdf5_episode(args.episode)
source = json.loads((args.native / "report.json").read_text())
if digest(args.native / "trajectory.hdf5") != source["trace_sha256"]:
    raise ValueError("示范 trajectory SHA256 不一致")
if (source["task"], source["task_profile"]) != (meta.task_id, meta.task_profile):
    raise ValueError("模型与示范的任务或 profile 不一致")
with h5py.File(args.episode) as stream:
    if not bool(stream.attrs["success"]) or stream.attrs["task_profile"] != meta.task_profile:
        raise ValueError("初始状态需要来自完整成功的原生 episode")
    if "sim/environment_origin" not in stream:
        raise ValueError("初始状态需要记录所属环境的原点")
with h5py.File(args.native / "trajectory.hdf5") as stream:
    expected_observation = stream["policy_observation"][0, 0]
git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
validator_sha256 = digest(Path(__file__))
args.task, args.task_profile = meta.task_id, meta.task_profile
args.num_envs, args.seed = 1, meta.config.seed
args.environment_mode, args.reward_discount = meta.config.environment_mode, meta.config.gamma
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    import torch
    from openso101.teleop.sim_state import _replay_restore_sim_state_from_episode
    from openso101.rl.backends import get_backend
    from openso101.rl.execution import build_environment
    from openso101.rl.vision_distillation import action_mapping
    from openso101.teleop.replay_validation import native_replay_recorder

    args.recorder_cfg = native_replay_recorder()
    env = build_environment(args, training=False)
    policy = get_backend(meta.config.backend).load(env, args.checkpoint)
    observation, _ = env.reset()
    runtime = env.unwrapped
    with h5py.File(args.episode) as stream:
        _replay_restore_sim_state_from_episode(runtime, runtime.scene, stream, 0)
    runtime.action_manager.reset()
    observation = runtime.observation_manager.compute()
    error = float(np.abs(observation["policy"][0].cpu().numpy() - expected_observation).max())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".initial_observation.json").open("x") as stream:
        json.dump({"maximum_error": error, "expected": expected_observation.tolist(),
                   "actual": observation["policy"][0].cpu().tolist(),
                   "source_episode_sha256": digest(args.episode), "source_trace_sha256": source["trace_sha256"]}, stream, indent=2)
    if error > 1e-6:
        raise ValueError(f"恢复后的实际初始 policy 观测与示范不一致: {error}")
    mapping = action_mapping(runtime)
    if args.arm_target_speed_limit is not None:
        arm_mapping = [item for item in mapping if item["joint_name"] != "Jaw"]
        if len(arm_mapping) != 5 or any(item["type"] != "position" for item in arm_mapping):
            raise ValueError("arm 目标速度检查需要五个归一化位置控制关节")
        arm_indices = [item["action_index"] for item in arm_mapping]
        scales = torch.tensor([item["scale"] for item in arm_mapping], device=runtime.device)
        offsets = torch.tensor([item["offset"] for item in arm_mapping], device=runtime.device)
        robot = runtime.scene["robot"]
        joints = [robot.joint_names.index(item["joint_name"]) for item in arm_mapping]
        previous_arm = (robot.data.joint_pos[:, joints] - offsets) / scales
        maximum_delta = args.arm_target_speed_limit * runtime.step_dt / scales
    trajectory, success, task_return = [], False, 0.
    with torch.inference_mode():
        while app.is_running():
            policy_observation = observation["policy"][0].detach().cpu().tolist()
            model_action = policy(observation)
            action = model_action.clone()
            if args.arm_target_speed_limit is not None:
                arm = previous_arm + torch.maximum(torch.minimum(action[:, arm_indices] - previous_arm,
                                                                  maximum_delta), -maximum_delta)
                action[:, arm_indices] = arm
                previous_arm = arm
            observation, reward, terminated, truncated, _ = env.step(action)
            task_return += float(reward[0])
            transition = runtime._replay_transition
            trajectory.append({"policy_observation": policy_observation,
                               "model_action": model_action[0].cpu().tolist(),
                               "action": action[0].cpu().tolist(), "reward": float(reward[0]),
                               "joint_targets": transition["targets"].cpu().tolist(),
                               "joint_position": transition["joint_position"].cpu().tolist(),
                               "joint_velocity": transition["joint_velocity"].cpu().tolist(),
                               "object_position_root": transition["object_position_root"].cpu().tolist(),
                               "jaw_forces": transition["jaw_forces"].cpu().tolist(),
                               "hold_seconds": float(transition["hold_seconds"]),
                               "success": transition["success"]})
            if bool((terminated | truncated)[0]):
                success = runtime._replay_transition["success"]
                break
    if not trajectory or not bool((terminated | truncated)[0]):
        raise RuntimeError("相同初始状态的策略检查提前结束")
    result = {"scope": "recorded_initial_state", "task": meta.task_id, "task_profile": meta.task_profile,
              "success": success, "steps": len(trajectory), "return": task_return,
              "initial_observation_error": error, "trajectory": trajectory,
              "source_episode_sha256": digest(args.episode), "source_trace_sha256": source["trace_sha256"],
              "model_sha256": digest(args.checkpoint / meta.checkpoint), "evaluation_git_sha": git_sha,
              "arm_target_speed_limit_rad_s": args.arm_target_speed_limit,
              "independent_success_rate_verified": False, "validator_sha256": validator_sha256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "trajectory"}), flush=True)
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
