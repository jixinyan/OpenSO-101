import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openso101.rl.gpu_guard import gpu_inventory, idle_device
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
device = next(item for item in gpu_inventory() if item["index"] == args.gpu)
if idle_device(device):
    raise RuntimeError("占用检查需要实际有作业运行的设备")
repo = Path.cwd().resolve()
worker_output = args.output / "blocked_worker"
receipt = args.output / "guard.json"
environment = os.environ | {"OPENSO101_REPO": str(repo), "CUDA_VISIBLE_DEVICES": ""}
with (args.output / "guard.log").open("x") as stream:
    result = subprocess.run([sys.executable, "scripts/run_idle_gpu.py", "--gpu", str(args.gpu),
                             "--report", str(receipt), "--", sys.executable,
                             "scripts/run_cpu_checks.py", "--output", str(worker_output)],
                            env=environment, stdout=stream, stderr=subprocess.STDOUT)
content = json.loads(receipt.read_text())
if result.returncode != 75 or content["status"] != "device_busy" or content["gpu_job_started"] or worker_output.exists():
    raise RuntimeError("实际已占用 GPU 的启动阻止检查未通过")
report = {"status": "occupied_gpu_launch_block_verified", "physical_gpu": args.gpu,
          "actual_occupants": content["initial_device"]["processes"], "worker_started": False,
          "gpu_context_created": False, "guard_report_sha256": file_digest(receipt),
          "source_sha256": file_digest(Path("src/openso101/rl/gpu_guard.py"))}
(args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False))
