import argparse
from argparse import Namespace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from openso101.rl.config import CheckpointMeta, TrainCfg, digest
from openso101.rl.snapshot import snapshot


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
config = TrainCfg(iterations=1000, rollout_steps=32, hidden_dims=(32, 16), environment_mode="nominal",
                  action_distribution="tanh_gaussian")
configuration = output / "requested_config.json"
configuration.write_text(config.model_dump_json(indent=2))
run = output / "run"
command = [sys.executable, "-u", "-m", "openso101.cli.main", "rl", "train", "--task", "OpenSO101-Lift-v0",
           "--backend", "rsl_rl", "--algo", "ppo", "--task-profile", "grasp_v4",
           "--train-config", str(configuration), "--num_envs", "4", "--output", str(run),
           "--headless", "--no-video", "--logger", "tensorboard"]
with (output / "training.log").open("x") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    try:
        checkpoint = run / "model_1.pt"
        while not (run / "model_2.pt").exists():
            if process.poll() is not None:
                raise RuntimeError(f"停止检查的实际训练提前终止：exit_code={process.returncode}")
            if time.monotonic() - started > 300:
                raise TimeoutError("停止检查的实际训练没有在五分钟内生成模型")
            time.sleep(.5)
        preserved = output / "snapshot"
        snapshot(Namespace(run=str(run), checkpoint=checkpoint.name, output=str(preserved)))
        metadata = CheckpointMeta.read(preserved)
        if process.poll() is not None or os.getpgid(process.pid) != process.pid:
            raise RuntimeError("停止检查需要仍在运行的独立训练进程")
        stop_started = time.monotonic()
        os.killpg(process.pid, signal.SIGTERM)
        exit_code = process.wait(timeout=30)
        stopped_seconds = time.monotonic() - stop_started
        if exit_code not in (0, 128 + signal.SIGTERM):
            raise RuntimeError(f"训练停止的实际 exit_code 不一致：{exit_code}")
        stop_record = json.loads((run / "training_stop.json").read_text())
        if (stop_record["status"] != "stop_requested" or stop_record["signal"] != signal.SIGTERM
                or stop_record["worker_pid"] != process.pid or stop_record["training_git_sha"] != metadata.git_sha
                or (run / "checkpoint.json").exists()):
            raise RuntimeError("停止检查的实际 signal 来源与训练状态不一致")
        checked = CheckpointMeta.read(preserved)
        if checked != metadata or digest(checkpoint) != metadata.files[checkpoint.name]:
            raise RuntimeError("停止后的实际保存模型或副本发生变化")
        report = {"status": "native_training_sigterm_verified", "created_at": datetime.now(UTC).isoformat(),
                  "training_command": command, "worker_pid": process.pid, "signal": "SIGTERM",
                  "exit_code": exit_code, "stop_seconds": stopped_seconds,
                  "stop_record_sha256": digest(run / "training_stop.json"),
                  "training_completed": False,
                  "preserved_checkpoint_sha256": metadata.files[checkpoint.name],
                  "preserved_transitions": metadata.completed_transitions,
                  "training_git_sha": metadata.git_sha, "source_sha256": digest(Path(__file__)),
                  "all_original_models_retained": True, "task_success_verified": False}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
