import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import torch

from openso101.il.evaluation import CohortEvaluation
from openso101.il.observations import camera_to_policy
from openso101.il.policies import load_policy
from openso101.il.policies.inference import select_sim_action
from openso101.il.policies.validation import model_state_digest
from openso101.il.policies.simulation import load_simulation_settings, attach_simulation_settings, SIMULATION_SETTINGS_FILE
from openso101.scenes.models import file_digest
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode
from openso101.teleop.simulation import recorded_simulation
from openso101.teleop.so101_mapping import batched_action_to_motor_units, batched_motor_units_to_action


def recorded_statistics(source: Path) -> dict:
    report_path, trace_path = source / "report.json", source / "trajectory.hdf5"
    reference = json.loads(report_path.read_text())
    if file_digest(trace_path) != reference["trajectory_sha256"]:
        raise ValueError("IL 统计检查的实际并行轨迹 SHA256 不一致")
    with h5py.File(trace_path) as trace:
        successes = torch.from_numpy(trace["success"][:])
        terminated = torch.from_numpy(trace["terminated"][:])
    environments = len(reference["environments"])
    if successes.dtype != torch.bool or successes.shape != terminated.shape or successes.shape[1] != environments:
        raise ValueError("IL 统计检查需要实际逐环境结束记录")
    if not bool(terminated.any(dim=0).all()):
        raise ValueError("来源并行轨迹需要包含全部环境的 episode 结束")
    summaries = []
    for requested in (1, 3, 4, 5, 7):
        campaign = CohortEvaluation(requested, environments, "cpu")
        while not campaign.finished:
            campaign.begin_cohort()
            for index in range(len(successes)):
                campaign.consume(successes[index] & terminated[index], terminated[index],
                                 torch.zeros_like(terminated[index]),
                                 torch.full((environments,), index + 1, dtype=torch.int64))
                if campaign.cohort_finished:
                    break
            if not campaign.cohort_finished:
                raise ValueError("实际并行轨迹尚未完成统计检查的批次")
        result = campaign.report()
        if result["n"] != requested or sum(result["episodes_per_environment"]) != requested:
            raise ValueError("IL 统计没有保存精确的 episode 数量")
        for item in result["episodes"]:
            expected = reference["environments"][item["environment"]]
            if item["success"] != expected["success"] or item["steps"] != expected["observed_steps"]:
                raise ValueError("IL 统计与实际 episode 结束步骤不一致")
        summaries.append(result)
    campaign = CohortEvaluation(environments, environments, "cpu")
    campaign.begin_cohort()
    campaign.consume(successes[0] & terminated[0], terminated[0], torch.zeros_like(terminated[0]),
                     torch.ones(environments, dtype=torch.int64))
    try:
        campaign.begin_cohort()
    except ValueError:
        pass
    else:
        raise RuntimeError("IL 统计允许清除仍在运行的 episode")
    partial = campaign.report()
    if partial["n"] != 0 or partial["success_rate"] is not None:
        raise ValueError("IL 中断统计需要保留尚未完成的 episode 状态")
    return {"source_report_sha256": file_digest(report_path), "source_trace_sha256": file_digest(trace_path),
            "source_environments": environments, "source_control_steps": successes.shape[0],
            "source_episode_steps": [item["observed_steps"] for item in reference["environments"]],
            "statistics_playbacks": summaries, "unfinished_cohort_reset_rejected": True,
            "partial_report": partial, "independent_task_success_verified": False}


def policy_actions(checkpoint: Path, episode: Path) -> dict:
    policy = load_policy(checkpoint, device="cpu")
    sources = {str(path.relative_to(checkpoint)): file_digest(path) for path in checkpoint.rglob("*") if path.is_file()}
    state_digest = model_state_digest(policy)
    observations = []
    with h5py.File(episode) as stream:
        source_fps = int(stream.attrs["fps"])
        settings = load_simulation_settings(checkpoint, source_fps)
        if settings.recorded_simulation != recorded_simulation(stream.attrs):
            raise ValueError("IL 模型保存的来源物理参数与实际 episode 不一致")
        for step in range(16):
            indices = list(range(step * 3, step * 3 + 3))
            positions = torch.from_numpy(stream["observations/qpos"][indices]).float()
            current = {"observation.state": batched_action_to_motor_units(positions)}
            for name in ("wrist_camera", "overhead_camera"):
                current[f"observation.images.{name}"] = camera_to_policy(
                    torch.from_numpy(stream[f"observations/images/{name}"][indices]))
            observations.append(current)
    if settings.fps != source_fps or any(tuple(observations[0][f"observation.images.{name}"].shape[2:]) != shape
                                         for name, shape in settings.camera_sizes.items()):
        raise ValueError("IL 保存的频率与相机尺寸需要符合实际采集数据")
    def actions(shared: bool):
        values = []
        with torch.inference_mode():
            for start in (0, 8):
                policy.reset()
                torch.manual_seed(42 + start)
                for current in observations[start:start + 8]:
                    if shared:
                        action = select_sim_action(policy, current, "cpu", 3)
                    else:
                        processed = policy.openso101_preprocessor(current)
                        motors = policy.openso101_postprocessor(policy.select_action(processed))
                        action = batched_motor_units_to_action(motors)
                    if action.shape != (3, 6) or action.device.type != "cpu" or not torch.isfinite(action).all():
                        raise ValueError("完整模型的 IL 并行动作格式无效")
                    values.append(action.clone())
        return torch.stack(values)
    expected, observed = actions(False), actions(True)
    maximum_error = float((expected - observed).abs().max())
    if maximum_error > 1e-6 or model_state_digest(policy) != state_digest:
        raise ValueError("共享 IL 推理与实际 LeRobot 模型序列推理不一致")
    if {str(path.relative_to(checkpoint)): file_digest(path) for path in checkpoint.rglob("*") if path.is_file()} != sources:
        raise ValueError("IL 推理检查期间模型文件内容发生变化")
    return {"policy": policy.config.type, "model_state_sha256": state_digest, "checkpoint_files": sources,
            "observation_frames": 48, "environments": 3, "cohorts": 2, "select_action_calls": 16,
            "maximum_action_error_rad": maximum_error, "model_files_unchanged": True,
            "simulation_settings": settings.model_dump(mode="json"),
            "optimizer_updates": 0, "task_success_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("IL 评估检查需要禁止 CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    validate_hdf5_episode(args.episode)
    episode_digest = file_digest(args.episode)
    statistics = recorded_statistics(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    policies = {}
    for name in ("act", "diffusion"):
        print(f"检查实际 {name} 并行动作与批次重置", flush=True)
        result = policy_actions(args.checkpoints / name / "pretrained_model", args.episode)
        policies[name] = result
        (args.output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    rejected_requests = []
    checkpoint = args.checkpoints / "act/pretrained_model"
    requests = [([command, "--task", "OpenSO101-Lift-v0"], "采集任务不一致") for command in ("play", "eval")]
    requests.append((["eval", "--task", "OpenSO101-PickPlace-v0", "--output", str(checkpoint / "evaluation.json")],
                     "模型目录之外"))
    for arguments, expected_message in requests:
        result = subprocess.run([sys.executable, "-m", "openso101.cli.main", "il", *arguments,
                                 "--policy-path", str(checkpoint)], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or expected_message not in result.stderr or "AppLauncher" in result.stderr:
            raise RuntimeError("实际 IL 入口需要在启动 Isaac 前拒绝来源参数或报告路径不一致的请求")
        rejected_requests.append({"arguments": arguments, "native_startup_prevented": True})
    if (checkpoint / "evaluation.json").exists():
        raise ValueError("IL 输入检查期间创建了模型目录中的评估报告")
    runtime_settings = args.checkpoints / "act/pretrained_model" / SIMULATION_SETTINGS_FILE
    if runtime_settings.is_file():
        checkpoint = args.output / "checkpoint_settings/checkpoints/last/pretrained_model"
        checkpoint.mkdir(parents=True)
        source_checkpoint = args.checkpoints / "act/pretrained_model"
        for path in source_checkpoint.iterdir():
            if path.is_file() and path.name != SIMULATION_SETTINGS_FILE:
                (checkpoint / path.name).hardlink_to(path)
        attached = attach_simulation_settings(args.output / "checkpoint_settings", runtime_settings)
        settings = load_simulation_settings(checkpoint)
        if len(attached) != 1 or settings.model_dump(mode="json") != policies["act"]["simulation_settings"]:
            raise ValueError("训练 checkpoint 的仿真设置保存与读取不一致")
        try:
            load_simulation_settings(checkpoint, settings.fps + 1)
        except ValueError:
            pass
        else:
            raise RuntimeError("IL 模型允许使用不同的采集频率")
        runtime_settings_verified = True
    else:
        runtime_settings_verified = False
    if file_digest(args.episode) != episode_digest or torch.cuda.is_initialized():
        raise ValueError("IL 评估检查需要保留原始 episode，并保持 CUDA 未初始化")
    result = {"status": "actual_recorded_il_evaluation_verified", "statistics": statistics, "policies": policies,
              "source_episode_sha256": episode_digest, "source_unchanged": True,
              "checkpoint_simulation_settings_verified": runtime_settings_verified,
              "recorded_simulation_preserved": True, "rejected_requests": rejected_requests,
              "source_files": {path: file_digest(Path(path)) for path in (
                  "src/openso101/il/evaluation.py", "src/openso101/il/policies/inference.py",
                  "src/openso101/il/policies/simulation.py",
                  "src/openso101/il/runners/evaluator.py", "src/openso101/teleop/success.py",
                  "src/openso101/teleop/simulation.py")},
              "validator_sha256": file_digest(Path(__file__)), "gpu_tests_started": False,
              "native_physics_verified": False, "new_policy_task_success_verified": False,
              "training_started": False}
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "gpu_tests_started", "training_started")}), flush=True)


if __name__ == "__main__":
    main()
