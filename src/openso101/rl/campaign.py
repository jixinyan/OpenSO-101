import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import CheckpointMeta, TrainCfg, digest
from .snapshot import snapshot


def campaign(args):
    config_path = Path(args.train_config).resolve()
    config = TrainCfg.model_validate_json(config_path.read_text())
    if config.backend != "rsl_rl" or config.algo != "ppo":
        raise ValueError("当前 campaign 使用 rsl_rl PPO")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("campaign 需要三个独立 seed")
    if not 1 <= len(args.gpus) <= 6 or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        raise ValueError("campaign 需要一到六个独立 GPU 编号")
    if config.evaluation_episodes != 100 or config.evaluation_interval != 100:
        raise ValueError("campaign 需要每 100 iterations 评估 100 episodes")
    config.batch_size(args.num_envs)
    profile = args.task_profile
    if args.validate_loop:
        for name in ("robot_model", "collision_bundle", "mujoco_python"):
            value = getattr(args, name)
            if value is None or not Path(value).exists():
                raise ValueError(f"完整闭环需要有效的 {name}")
        if args.distillation_iterations <= 0:
            raise ValueError("完整闭环需要有效的 distillation_iterations")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tasks = ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0")
    jobs = []
    handles = []
    processes = []
    receipt = {"schema_version": 2, "training_git_sha": revision, "config_sha256": digest(config_path),
               "created_at": datetime.now(UTC).isoformat(), "task_profile": profile,
               "environment_mode": config.environment_mode, "num_envs": args.num_envs,
               "required_seeds": args.seeds, "jobs": jobs, "multi_seed_acceptance_verified": False,
               "validation_loop_requested": args.validate_loop, "multi_seed_loop_verified": False}

    def write_receipt():
        (root / "campaign.json").write_text(json.dumps(receipt, indent=2) + "\n")

    write_receipt()
    try:
        for task, seed in ((task, seed) for seed in args.seeds for task in tasks):
            folder = root / f"{task}_seed_{seed}"
            command = [sys.executable, "-m", "openso101.cli.main", "rl", "train", "--task", task,
                       "--algo", "ppo", "--backend", "rsl_rl", "--train-config", str(config_path),
                       "--task-profile", profile, "--seed", str(seed), "--num_envs", str(args.num_envs),
                       "--output", str(folder), "--headless", "--no-video", "--logger", "tensorboard"]
            log_path = root / f"{folder.name}.log"
            jobs.append({"task": task, "seed": seed, "gpu": None, "pid": None,
                         "run": str(folder), "log": str(log_path), "command": command, "status": "queued"})
            write_receipt()
        while any(job["status"] in ("queued", "running") for job in jobs):
            busy = {job["gpu"] for job in jobs if job["status"] == "running"}
            for gpu in args.gpus:
                pending = next((job for job in jobs if job["status"] == "queued"), None)
                if gpu in busy or pending is None:
                    continue
                handle = Path(pending["log"]).open("x")
                handles.append(handle)
                process = subprocess.Popen(pending["command"],
                                           env=os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)},
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((process, pending))
                pending.update(gpu=gpu, pid=process.pid, status="running",
                               started_at=datetime.now(UTC).isoformat())
                write_receipt()
            for process, job in processes:
                code = process.poll()
                if code is None or job["status"] != "running":
                    continue
                job["exit_code"] = code
                folder = Path(job["run"])
                if code != 0 or not (folder / "checkpoint.json").is_file():
                    job["status"] = "failed"
                    write_receipt()
                    raise RuntimeError(f"训练未生成完整 checkpoint：{folder}")
                meta = CheckpointMeta.read(folder)
                convergence = json.loads((folder / "convergence.json").read_text())
                history = json.loads((folder / "evaluation_history.json").read_text())
                accepted = (len(history) >= 3 and all(
                    len(item["episodes"]) == 100 and sum(record["success"] for record in item["episodes"]) >= 90
                    for item in history[-3:]))
                if accepted != convergence["seed_converged"] or convergence["seed"] != job["seed"]:
                    raise RuntimeError(f"训练与独立评估记录不一致：{folder}")
                from argparse import Namespace

                best_folder = folder / "selected_policy"
                snapshot(Namespace(run=str(folder), checkpoint="model_best.pt", output=str(best_folder)))
                job.update(status="accepted" if accepted else "completed_below_threshold",
                           seed_converged=accepted, checkpoint_sha256=meta.files[meta.checkpoint],
                           selected_policy=str(best_folder), completed_transitions=meta.completed_transitions)
                write_receipt()
                if accepted and args.validate_loop:
                    loop_folder = root / f"{folder.name}_validation_loop"
                    command = [sys.executable, "-u", "-m", "openso101.cli.main", "rl", "validate-loop",
                               "--teacher-run", str(best_folder), "--output", str(loop_folder),
                               "--robot-model", str(Path(args.robot_model).resolve()),
                               "--collision-bundle", str(Path(args.collision_bundle).resolve()),
                               "--mujoco-python", str(Path(args.mujoco_python).resolve()),
                               "--distillation-iterations", str(args.distillation_iterations),
                               "--seed", str(30000 + job["seed"])]
                    job.update(validation_loop_status="running", validation_loop_command=command)
                    write_receipt()
                    with (root / f"{folder.name}_validation_loop.log").open("x") as log:
                        completed = subprocess.run(command, env=os.environ | {"CUDA_VISIBLE_DEVICES": str(job["gpu"])},
                                                   stdout=log, stderr=subprocess.STDOUT)
                    report = loop_folder / "validation_loop.json"
                    job.update(validation_loop_exit_code=completed.returncode,
                               validation_loop_report=str(report), validation_loop_report_sha256=digest(report))
                    loop_receipt = json.loads(report.read_text())
                    job["validation_loop_status"] = loop_receipt["status"]
                    job["single_policy_loop_verified"] = loop_receipt["single_policy_loop_verified"]
                    write_receipt()
                    if completed.returncode or not job["single_policy_loop_verified"]:
                        raise RuntimeError(f"独立闭环验收未通过：{loop_folder}")
                print(json.dumps(job), flush=True)
            time.sleep(10)
        receipt["multi_seed_acceptance_verified"] = all(job["status"] == "accepted" for job in jobs)
        receipt["multi_seed_loop_verified"] = receipt["multi_seed_acceptance_verified"] and all(
            job.get("single_policy_loop_verified", False) for job in jobs)
        receipt["completed_at"] = datetime.now(UTC).isoformat()
        write_receipt()
        return 0 if receipt["multi_seed_acceptance_verified"] else 1
    finally:
        for process, job in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                job["status"] = "terminated"
        for process, job in processes:
            process.wait()
        for handle in handles:
            handle.close()
        write_receipt()
