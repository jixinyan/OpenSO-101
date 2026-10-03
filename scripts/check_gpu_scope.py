import argparse
import csv
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import subprocess

import psutil

from openso101.rl.config import digest
from openso101.rl.gpu_scope import gpu_scope


parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists() or len(set(args.pid)) != len(args.pid):
    raise ValueError("GPU 检查需要新的报告文件与唯一的进程 identifier")
scope = gpu_scope()
inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
gpu_indices = {row[1].strip(): int(row[0]) for row in csv.reader(io.StringIO(inventory))}
computing = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
allocations = {}
for row in csv.reader(io.StringIO(computing)):
    allocations.setdefault(int(row[0]), set()).add(gpu_indices[row[1].strip()])
records = []
for pid in args.pid:
    process = psutil.Process(pid)
    command = process.cmdline()
    cwd = Path(process.cwd()).resolve()
    if not cwd.name.startswith("OpenSO-101"):
        raise ValueError("指定进程需要属于 OpenSO-101 工作目录")
    visible = process.environ()["CUDA_VISIBLE_DEVICES"]
    devices = [int(value) for value in next(csv.reader([visible]))]
    scope.validate_allocation(devices)
    actual = sorted(allocations.get(pid, set()))
    if actual:
        scope.validate_allocation(actual)
        if set(actual) != set(devices):
            raise ValueError("实际 GPU 与进程指定的 GPU 不一致")
    records.append({"pid": pid, "command": command, "cwd": str(cwd),
                    "requested_gpus": devices, "actual_gpus": actual})
result = {"status": "actual_gpu_scope_verified", "created_at": datetime.now(UTC).isoformat(),
          "allowed_gpus": scope.allowed_gpus, "processes": records, "source_sha256": digest(Path(__file__))}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
