import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal

import psutil

from openso101.rl.config import CheckpointMeta, digest


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--parent-stopped", action="store_true")
parser.add_argument("--force-workers", action="store_true")
args = parser.parse_args()
folder = args.folder.resolve()
receipt_path = folder / "campaign.json"
receipt = json.loads(receipt_path.read_text())
metadata = CheckpointMeta.read(args.snapshot)
if metadata.git_sha != receipt["training_git_sha"]:
    raise ValueError("保存模型的来源与待终止训练不一致")
if args.parent_stopped:
    if psutil.pid_exists(args.pid):
        raise ValueError("parent-stopped 需要 campaign 进程已经终止")
    parent = None
else:
    parent = psutil.Process(args.pid)
    command = parent.cmdline()
    if command[1:5] != ["-u", "-m", "openso101.cli.main", "rl"] or "campaign" not in command:
        raise ValueError("指定进程未通过 campaign 检查")
    if "--output" not in command or (Path(parent.cwd()) / command[command.index("--output") + 1]).resolve() != folder:
        raise ValueError("campaign 进程与指定训练目录不一致")
workers = []
for job in receipt["jobs"]:
    if job["status"] == "running":
        worker = psutil.Process(job["pid"])
        if worker.cmdline() != job["command"] or os.getpgid(worker.pid) != worker.pid:
            raise ValueError("训练进程及其 process group 未通过来源检查")
        workers.append((worker, job))
if len(workers) != 1 or metadata.task_id != workers[0][1]["task"] or metadata.config.seed != workers[0][1]["seed"]:
    raise ValueError("终止检查需要与当前单个训练任务一致的模型副本")
if parent is not None:
    parent.terminate()
    parent.wait(timeout=10)
for worker, job in workers:
    os.killpg(worker.pid, signal.SIGKILL if args.force_workers else signal.SIGTERM)
    worker.wait(timeout=30)
    job["status"] = "terminated"
for job in receipt["jobs"]:
    if job["status"] == "queued":
        job["status"] = "cancelled"
receipt["stopped_at"] = datetime.now(UTC).isoformat()
receipt["preserved_snapshot"] = str(args.snapshot.resolve())
receipt["preserved_checkpoint_sha256"] = metadata.files[metadata.checkpoint]
receipt["stop_source_sha256"] = digest(Path(__file__))
receipt["worker_stop_signal"] = "SIGKILL" if args.force_workers else "SIGTERM"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": "campaign_stopped", "folder": str(folder),
                  "preserved_snapshot": receipt["preserved_snapshot"],
                  "preserved_checkpoint_sha256": receipt["preserved_checkpoint_sha256"]}), flush=True)
