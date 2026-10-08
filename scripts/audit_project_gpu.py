import argparse
import getpass
import json
from datetime import UTC, datetime
from pathlib import Path

import psutil

from openso101.rl.gpu_guard import gpu_inventory
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--stop", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
repo = Path(__file__).resolve().parents[1]
username = getpass.getuser()
inventory = gpu_inventory()
workers = {}
for device in inventory:
    for item in device["processes"]:
        if item["owner"] != username or item["pid"] in workers:
            continue
        process = psutil.Process(item["pid"])
        cwd = Path(process.cwd()).resolve()
        command = process.cmdline()
        script_owned = any((cwd / value).resolve().is_relative_to(repo / "scripts") for value in command[1:]
                           if not value.startswith("-"))
        if cwd.is_relative_to(repo) and ("openso101.cli.main" in command or script_owned):
            workers[item["pid"]] = process
records = [{"pid": pid, "create_time": process.create_time(), "cwd": process.cwd(),
            "command": process.cmdline()} for pid, process in workers.items()]
if args.stop:
    for process in workers.values():
        process.terminate()
    _, alive = psutil.wait_procs(list(workers.values()), timeout=30)
    if alive:
        raise RuntimeError("本项目的 GPU 进程尚未终止")
    remaining = gpu_inventory()
    if any(item["pid"] in workers for device in remaining for item in device["processes"]):
        raise RuntimeError("本项目终止后的 GPU 上下文仍然存在")
else:
    remaining = inventory
report = {"status": "project_gpu_processes_stopped" if records and args.stop else (
    "project_gpu_processes_absent" if not records else "project_gpu_processes_running"),
    "checked_at": datetime.now(UTC).isoformat(), "stop_requested": args.stop,
    "project_workers": records, "remaining_inventory": remaining,
    "source_sha256": file_digest(Path(__file__)), "other_projects_terminated": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": report["status"], "project_gpu_process_count": len(records)}))
