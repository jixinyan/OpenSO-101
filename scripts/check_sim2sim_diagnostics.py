import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--evaluation", type=Path, required=True)
parser.add_argument("--diagnostics", type=Path, help="检查已有诊断目录；省略时运行实际 CPU 诊断")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OPENSO101_SKIP_ISAAC") != "1":
    raise RuntimeError("检查需要禁用 CUDA 与原生 Isaac 启动")
args.output.mkdir(parents=True, exist_ok=False)
created = args.diagnostics is None
if created:
    args.diagnostics = args.output / "diagnostics"
    command = [sys.executable, "-m", "openso101.cli.main", "sim2sim", "diagnose", "--policy", str(args.policy),
               "--robot-model", str(args.robot_model), "--evaluation", str(args.evaluation),
               "--output", str(args.diagnostics)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (args.output / "diagnostics.stdout.log").write_text(result.stdout)
    (args.output / "diagnostics.stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"实际 CPU 诊断未完成：{args.output / 'diagnostics.stderr.log'}")
diagnostics_path = args.diagnostics / "report.json"
diagnostics = json.loads(diagnostics_path.read_text())
if diagnostics["status"] != "actual_sim2sim_diagnostics_verified" or diagnostics["gpu_tests_started"]:
    raise ValueError("需要已完成的实际 CPU 轨迹诊断")
if digest(Path("src/openso101/sim2sim/diagnostics.py")) != diagnostics["source_sha256"]:
    raise ValueError("诊断报告与当前源码 SHA256 不一致")
sources = [args.policy / "policy.json", args.policy / "policy.pt", args.policy / "validation.json",
           args.policy / "isaac_validation.hdf5", args.robot_model, args.evaluation / "trajectory.hdf5",
           args.evaluation / "report.json", diagnostics_path, args.diagnostics / "diagnostics.png"]
before = {str(path): digest(path) for path in sources}
if before[str(args.diagnostics / "diagnostics.png")] != diagnostics["figure_sha256"]:
    raise ValueError("实际曲线 SHA256 不一致")
with Image.open(args.diagnostics / "diagnostics.png") as image:
    image.load()
    dimensions = image.size
    if image.format != "PNG" or dimensions[0] < 1000 or dimensions[1] < 500:
        raise ValueError("实际曲线的编码或尺寸无效")
    if float(np.asarray(image.convert("RGB"), dtype=np.float64).std()) <= 1:
        raise ValueError("实际曲线没有有效像素变化")
records = []


def reject(name, evaluation, output, expected):
    command = [sys.executable, "-m", "openso101.cli.main", "sim2sim", "diagnose", "--policy", str(args.policy),
               "--robot-model", str(args.robot_model), "--evaluation", str(evaluation), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout = args.output / f"{name}.stdout.log"
    stderr = args.output / f"{name}.stderr.log"
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    if result.returncode == 0 or expected not in result.stderr:
        raise RuntimeError(f"实际无效输入未被拒绝：{name}")
    if output != args.diagnostics and output.exists():
        raise RuntimeError(f"输入检查创建了输出目录：{name}")
    records.append({"name": name, "exit_code": result.returncode, "expected_error": expected,
                    "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr)})


for name in ("report_hash", "action_targets", "field_length", "initial_state"):
    evaluation = args.output / name
    evaluation.mkdir()
    shutil.copy2(args.evaluation / "trajectory.hdf5", evaluation / "trajectory.hdf5")
    report = json.loads((args.evaluation / "report.json").read_text())
    if name == "report_hash":
        report["policy_metadata_sha256"] = "0" * 64
        expected = "diagnostics 来源 SHA256 不一致：policy_metadata_sha256"
    else:
        with h5py.File(evaluation / "trajectory.hdf5", "r+") as stream:
            group = stream["episode_000000"]
            if name == "action_targets":
                group["joint_targets"][0, 0] += .05
                expected = "实际 action 与保存的 joint_targets 不一致"
            elif name == "initial_state":
                group["joint_position"][0, 0] += .05
                expected = "Isaac 与 MuJoCo 的实际初始状态不一致"
            else:
                forces = group["jaw_forces"][:]
                del group["jaw_forces"]
                group.create_dataset("jaw_forces", data=np.concatenate((forces, forces[-1:])))
                expected = "实际轨迹的字段形状无效：jaw_forces"
        report["trajectory_sha256"] = digest(evaluation / "trajectory.hdf5")
    (evaluation / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    reject(name, evaluation, args.output / f"{name}_diagnostics", expected)
reject("existing_output", args.evaluation, args.diagnostics, "FileExistsError")
reject("source_overlap", args.evaluation, args.evaluation / "new_diagnostics", "输出目录需要与来源目录独立")
after = {str(path): digest(path) for path in sources}
if before != after:
    raise RuntimeError("诊断检查改变了实际来源或既有报告")
report = {"status": "actual_cpu_sim2sim_diagnostics_verified", "process_checks": records,
          "process_check_count": len(records),
          "diagnostics_created": created,
          "environments": len(diagnostics["environments"]), "source_unchanged": True,
          "diagnostics_sha256": digest(diagnostics_path), "figure_dimensions": dimensions,
          "figure_sha256": diagnostics["figure_sha256"], "inputs": before,
          "source_sha256": digest(Path(__file__)), "gpu_tests_started": False,
          "training_started": False, "rl_task_success_verified": False}
(args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False), flush=True)
