import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from openso101.rl.config import digest
from openso101.rl.gpu_scope import configure_visible_gpu


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
configure_visible_gpu()
source = json.loads((args.native / "report.json").read_text())
if digest(args.native / "trajectory.hdf5") != source["trace_sha256"]:
    raise ValueError("来源 trajectory SHA256 不一致")
if digest(Path(f"src/openso101/tasks/shared/{source['task_profile']}.py")) != source["profile_sha256"]:
    raise ValueError("任务配置版本需要与实际采集一致")
with h5py.File(args.native / "trajectory.hdf5") as stream:
    observations = stream["policy_observation"][0]
    actions = stream["policy_action"][:]
    source_active = stream["active"][:]
    targets = stream["joint_targets"][:]
if any(not np.isfinite(value).all() for value in (observations, actions, targets)):
    raise ValueError("实际采集记录需要有限数值")
if actions.shape[:2] != source_active.shape or actions.shape[1] != source["episodes"]:
    raise ValueError("完整并行轨迹的步骤与环境数量不一致")
args.task, args.task_profile, args.seed = source["task"], source["task_profile"], source["seed"]
args.num_envs, args.environment_mode = source["episodes"], source["environment_mode"]
args.reward_discount = .99
args.output.mkdir(parents=True, exist_ok=False)
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
source_sha256 = digest(Path(__file__))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    import torch
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils import configclass
    from isaaclab.utils.math import subtract_frame_transforms

    from openso101.rl.execution import build_environment
    from openso101.rl.vision_distillation import action_mapping
    from openso101.robots import SO101_SIM_JOINT_NAMES
    from openso101.tasks.shared.grasp import _jaw_force_magnitude

    trajectory = []

    class CohortRecorder(RecorderTerm):
        def record_post_step(self):
            runtime = self._env
            robot, obj = runtime.scene["robot"], runtime.scene["object"]
            ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
            position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                    obj.data.root_pos_w, obj.data.root_quat_w)
            hold = (runtime.command_manager.get_term("object_pose").placement_hold_seconds
                    if runtime.cfg.task_profile_task == "pick_place"
                    else runtime.termination_manager.get_term_cfg("success").func.hold_seconds)
            values = {"joint_position": robot.data.joint_pos[:, ids],
                      "joint_velocity": robot.data.joint_vel[:, ids], "object_position_root": position,
                      "targets": torch.cat([runtime.action_manager.get_term(name).processed_actions
                                            for name in runtime.action_manager.active_terms], dim=-1),
                      "jaw_forces": torch.stack([_jaw_force_magnitude(runtime.scene[name])
                                                 for name in ("gripper_jaw_contact", "moving_jaw_contact")], dim=-1),
                      "success": runtime.termination_manager.get_term("success"),
                      "terminated": runtime.reset_terminated | runtime.reset_time_outs,
                      "hold_seconds": hold}
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise RuntimeError("实际并行回放产生无效数值")
            trajectory.append({name: value.detach().cpu().numpy().copy() for name, value in values.items()})
            return None, None

    @configclass
    class CohortRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        dataset_export_dir_path = str(args.output / "recorder")
        transition = RecorderTermCfg(class_type=CohortRecorder)

    args.recorder_cfg = CohortRecorderCfg()
    env = build_environment(args, training=True)
    actual_observations, _ = env.reset()
    runtime = env.unwrapped
    if runtime.step_dt != source["control_dt"] or runtime.physics_dt != source["physics_dt"]:
        raise ValueError("回放物理与控制周期需要与来源一致")
    if action_mapping(runtime) != source["policy_action_mapping"]:
        raise ValueError("回放动作映射需要与来源一致")
    initial_error = float(np.abs(actual_observations["policy"].cpu().numpy() - observations).max())
    if initial_error > 1e-6:
        raise ValueError(f"并行回放的初始观测需要与来源一致: {initial_error}")
    for action in actions:
        env.step(torch.as_tensor(action, device=runtime.device))
    if len(trajectory) != len(actions):
        raise RuntimeError("实际并行回放步骤不完整")
    arrays = {name: np.stack([item[name] for item in trajectory]) for name in trajectory[0]}
    target_error = float(np.abs(arrays["targets"] - targets).max())
    if target_error > 1e-6:
        raise ValueError("实际执行关节目标与来源不一致")
    with h5py.File(args.output / "trajectory.hdf5", "x") as stream:
        for name, value in arrays.items():
            stream.create_dataset(name, data=value)
    records = []
    for index in range(args.num_envs):
        selected = np.flatnonzero(source_active[:, index])
        terminations = np.flatnonzero(arrays["terminated"][selected, index])
        first_window = selected[:terminations[0] + 1] if len(terminations) else selected
        records.append({"environment": index, "source_success": source["environments"][index]["success"],
                        "source_steps": len(selected), "observed_steps": len(first_window),
                        "success": bool(arrays["success"][first_window, index].any()),
                        "bilateral_contact_steps": int((arrays["jaw_forces"][first_window, index] >= .5).all(-1).sum()),
                        "maximum_hold_seconds": float(arrays["hold_seconds"][first_window, index].max())})
    report = {"scope": "actual_complete_source_cohort_replay", "task": args.task, "task_profile": args.task_profile,
              "seed": args.seed, "environments": records, "successes": sum(item["success"] for item in records),
              "episodes": args.num_envs, "initial_observation_error": initial_error, "maximum_target_error": target_error,
              "source_trace_sha256": source["trace_sha256"], "source_report_sha256": digest(args.native / "report.json"),
              "trajectory_sha256": digest(args.output / "trajectory.hdf5"), "source_sha256": source_sha256,
              "evaluation_git_sha": git_sha, "single_environment_reproducibility_verified": False}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)
finally:
    if env is not None:
        env.close()
app.close()
