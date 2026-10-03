import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import CheckpointMeta, digest


def validate_loop(args):
    source = Path(args.teacher_run).resolve()
    metadata = CheckpointMeta.read(source)
    if metadata.task_id not in ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"):
        raise ValueError("闭环验证需要 Lift 或 PickPlace teacher")
    if metadata.config.backend != "rsl_rl" or metadata.config.algo != "ppo":
        raise ValueError("闭环验证需要 RSL PPO teacher")
    if args.distillation_iterations <= 0 or args.num_envs <= 0 or args.student_num_envs <= 0:
        raise ValueError("闭环验证需要有效的 iterations 和环境数量")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    teacher = root / "teacher"
    teacher.mkdir()
    for name in (*metadata.files, "checkpoint.json"):
        target = teacher / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    CheckpointMeta.read(teacher)
    receipt = {"schema_version": 1, "status": "running", "task": metadata.task_id,
               "task_profile": metadata.task_profile, "teacher_sha256": metadata.files[metadata.checkpoint],
               "source_metadata_sha256": digest(source / "checkpoint.json"),
               "validation_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "created_at": datetime.now(UTC).isoformat(), "episodes_per_evaluation": 100,
               "success_threshold": .9, "stages": [], "single_policy_loop_verified": False,
               "multi_seed_acceptance_verified": False, "hardware_run_verified": False}
    receipt_path = root / "validation_loop.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2))

    def execute(name, command):
        log = root / f"{name}.log"
        stage = {"name": name, "command": command, "started_at": datetime.now(UTC).isoformat(),
                 "status": "running", "log": str(log)}
        receipt["stages"].append(stage)
        save()
        with log.open("x") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        stage.update(exit_code=completed.returncode, completed_at=datetime.now(UTC).isoformat(),
                     log_sha256=digest(log), status="completed" if completed.returncode == 0 else "failed")
        if completed.returncode:
            receipt["status"] = "failed"
        save()
        if completed.returncode:
            raise RuntimeError(f"闭环验证步骤执行失败：{name}，日志：{log}")
        return stage

    def accept_task(stage, report_path):
        report = json.loads(report_path.read_text())
        episodes = report["episodes"]
        successes = sum(item["success"] is True for item in episodes)
        if len(episodes) != 100 or report["task"] != metadata.task_id or report["success_rate"] != successes / 100:
            receipt["status"] = "invalid_task_report"
            save()
            raise ValueError("闭环验证需要一致的 100 episodes 实际报告")
        stage.update(report=str(report_path), report_sha256=digest(report_path), successes=successes, episodes=100)
        if successes < 90:
            stage["status"] = "task_threshold_not_met"
            receipt["status"] = "task_threshold_not_met"
        save()
        if successes < 90:
            raise RuntimeError(f"{stage['name']} 的实际任务成功率为 {successes}/100，要求至少 90/100")
        return report

    def evaluation_report(folder):
        reports = list(folder.glob("evaluation-*.json"))
        if len(reports) != 1:
            raise RuntimeError("闭环验证需要唯一的本次独立评估报告")
        return reports[0]

    save()
    entry = [sys.executable, "-u", "-m", "openso101.cli.main"]
    stage = execute("teacher_evaluation", entry + ["rl", "eval", "--task", metadata.task_id,
                    "--checkpoint", str(teacher), "--num-envs", str(args.num_envs), "--n-episodes", "100",
                    "--seed", str(args.seed), "--headless"])
    teacher_report = accept_task(stage, evaluation_report(teacher))
    if teacher_report["checkpoint_sha256"] != receipt["teacher_sha256"]:
        raise ValueError("teacher 独立评估的模型 SHA256 不一致")
    portable = root / "portable"
    execute("portable_export", entry + ["rl", "export", "--task", metadata.task_id, "--checkpoint", str(teacher),
            "--output", str(portable), "--num-envs", "100", "--validation-steps", "300",
            "--seed", str(args.seed + 1), "--headless"])
    mujoco_output = root / "mujoco"
    stage = execute("mujoco_evaluation", [str(Path(args.mujoco_python).resolve()), "-u", "-m", "openso101.cli.main",
                    "sim2sim", "mujoco", "--policy", str(portable), "--robot-model", str(Path(args.robot_model).resolve()),
                    "--collision-bundle", str(Path(args.collision_bundle).resolve()), "--episodes", "100",
                    "--recorded-physics", "--constrained-drive", "--output", str(mujoco_output)])
    mujoco_report = accept_task(stage, mujoco_output / "report.json")
    if (not mujoco_report["actual_velocity_limits_verified"] or not all(
            item["constrained_drive"]["actual_torque_limits_verified"] for item in mujoco_report["episodes"])):
        raise RuntimeError("MuJoCo 的实际速度或力矩检查未通过")
    student = root / "student"
    execute("student_distillation", entry + ["rl", "distill", "--teacher-run", str(teacher), "--output", str(student),
            "--num-envs", str(args.student_num_envs), "--iterations", str(args.distillation_iterations),
            "--rollout-steps", "16", "--headless"])
    recording = root / "student_recording"
    stage = execute("student_evaluation", entry + ["rl", "student-eval", "--student", str(student),
                    "--num-envs", str(args.student_num_envs), "--n-episodes", "100", "--seed", str(args.seed + 2),
                    "--headless", "--recording-output", str(recording)])
    accept_task(stage, evaluation_report(student))
    execute("recorded_student_inference", entry + ["sim2real", "validate", "--policy-path", str(student),
            "--episode", str(recording / "episodes" / "episode_000000.hdf5"), "--output", str(root / "student_inference")])
    receipt.update(status="single_policy_loop_verified", single_policy_loop_verified=True,
                   completed_at=datetime.now(UTC).isoformat())
    save()
    print(json.dumps(receipt))
    return 0
