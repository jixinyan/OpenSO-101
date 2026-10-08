import argparse
import json
import os
from pathlib import Path

import torch

from openso101.rl.gpu_guard import launch_cuda_command, verify_cuda_inference
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or "OPENSO101_GPU_GUARD_PID" in os.environ:
    raise ValueError("CUDA 入口检查需要实际禁止 CUDA 的独立 CPU 进程")
if torch.cuda.is_initialized():
    raise ValueError("CUDA 入口检查不能初始化 GPU")
launch_cuda_command("cpu")
verify_cuda_inference("cpu")
rejected = []
for function, device in ((launch_cuda_command, "cuda:0"), (launch_cuda_command, "cuda:1"),
                         (verify_cuda_inference, "cuda:0"), (verify_cuda_inference, "cuda:1")):
    try:
        function(device)
    except ValueError:
        pass
    else:
        raise RuntimeError("禁止 CUDA 的进程需要拒绝 GPU 请求")
    rejected.append({"function": function.__name__, "device": device})
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA 入口检查不能初始化 GPU")
report = {"status": "actual_cpu_cuda_entry_preflight_verified", "rejected_requests": rejected,
          "cpu_inference_allowed": True, "gpu_tests_started": False, "training_started": False,
          "gpu_competition_verified": False, "hardware_run_verified": False,
          "source_sha256": file_digest(Path(__file__)),
          "guard_sha256": file_digest(Path("src/openso101/rl/gpu_guard.py"))}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(report, ensure_ascii=False))
