import json
import os
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
    if len(set(args.seeds)) != 3 or len(set(args.gpus)) != 6 or len(args.gpus) != 6:
        raise ValueError("campaign 需要三个独立 seed 和六个独立 GPU")
    if config.evaluation_episodes != 100 or config.evaluation_interval != 100:
        raise ValueError("campaign 需要每 100 iterations 评估 100 episodes")
    config.batch_size(args.num_envs)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tasks = ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0")
    jobs = []
    handles = []
    processes = []
    receipt = {"schema_version": 1, "training_git_sha": revision, "config_sha256": digest(config_path),
               "created_at": datetime.now(UTC).isoformat(), "task_profile": "grasp_v3",
               "environment_mode": config.environment_mode, "num_envs": args.num_envs,
               "required_seeds": args.seeds, "jobs": jobs, "multi_seed_acceptance_verified": False}

    def write_receipt():
        (root / "campaign.json").write_text(json.dumps(receipt, indent=2) + "\n")

    write_receipt()
    try:
        for index, (task, seed) in enumerate((task, seed) for task in tasks for seed in args.seeds):
            folder = root / f"{task}_seed_{seed}"
            command = [sys.executable, "-m", "openso101.cli.main", "rl", "train", "--task", task,
                       "--algo", "ppo", "--backend", "rsl_rl", "--train-config", str(config_path),
                       "--task-profile", "grasp_v3", "--seed", str(seed), "--num_envs", str(args.num_envs),
                       "--output", str(folder), "--headless", "--no-video", "--logger", "tensorboard"]
            environment = os.environ | {"CUDA_VISIBLE_DEVICES": str(args.gpus[index])}
            log_path = root / f"{folder.name}.log"
            handle = log_path.open("x")
            handles.append(handle)
            process = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            jobs.append({"task": task, "seed": seed, "gpu": args.gpus[index], "pid": process.pid,
                         "run": str(folder), "log": str(log_path), "command": command, "status": "running"})
            write_receipt()
        while any(job["status"] == "running" for job in jobs):
            for process, job in zip(processes, jobs, strict=True):
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
                print(json.dumps(job), flush=True)
            time.sleep(10)
        receipt["multi_seed_acceptance_verified"] = all(job["status"] == "accepted" for job in jobs)
        receipt["completed_at"] = datetime.now(UTC).isoformat()
        write_receipt()
        return 0 if receipt["multi_seed_acceptance_verified"] else 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()
