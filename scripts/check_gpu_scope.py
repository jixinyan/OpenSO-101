import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree

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
inventory = ElementTree.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
allocations = {}
allocation_details = {}
for gpu in inventory.findall("gpu"):
    index = int(gpu.findtext("minor_number"))
    for process in gpu.findall("processes/process_info"):
        pid = int(process.findtext("pid"))
        allocations.setdefault(pid, set()).add(index)
        allocation_details.setdefault(pid, []).append({
            "gpu": index, "type": process.findtext("type"),
            "used_memory": process.findtext("used_memory"), "gpu_uuid": gpu.findtext("uuid"),
        })
records = []
for pid in args.pid:
    process = psutil.Process(pid)
    command = process.cmdline()
    cwd = Path(process.cwd()).resolve()
    if not cwd.name.startswith("OpenSO-101"):
        raise ValueError("指定进程需要属于 OpenSO-101 工作目录")
    visible = process.environ()["CUDA_VISIBLE_DEVICES"]
    environment = process.environ()
    if environment.get("OPENSO101_GPU_NAMESPACE") == "1":
        devices = [int(environment["OPENSO101_PHYSICAL_GPU"])]
        if visible != environment["NVIDIA_VISIBLE_DEVICES"]:
            raise ValueError("隔离进程的 CUDA 与 NVIDIA UUID 必须一致")
    else:
        devices = [int(value) for value in next(csv.reader([visible]))]
    scope.validate_allocation(devices)
    actual = sorted(allocations.get(pid, set()))
    if actual:
        scope.validate_allocation(actual)
        if set(actual) != set(devices):
            raise ValueError("实际 GPU 与进程指定的 GPU 不一致")
    records.append({"pid": pid, "command": command, "cwd": str(cwd),
                    "requested_gpus": devices, "actual_gpus": actual,
                    "cuda_visible_devices": visible,
                    "device_namespace": environment.get("OPENSO101_GPU_NAMESPACE") == "1",
                    "allocations": allocation_details.get(pid, [])})
result = {"status": "actual_gpu_scope_verified", "created_at": datetime.now(UTC).isoformat(),
          "checked_process_types": ["compute", "graphics"],
          "allowed_gpus": scope.allowed_gpus, "processes": records, "source_sha256": digest(Path(__file__))}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
