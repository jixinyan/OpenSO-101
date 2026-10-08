import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openso101.il.datasets import load_lerobot_dataset
from openso101.il.policies.factory import load_policy
from openso101.il.policies.simulation import dataset_simulation, load_simulation_settings
from openso101.teleop.recorder.metadata import RECORDING_METADATA_FILE
from openso101.scenes.models import file_digest


def check_checkpoint_inputs(checkpoint: Path, output: Path):
    results = []
    state_file = next(path.name for path in checkpoint.glob("policy_preprocessor*.safetensors"))
    for missing in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json", state_file):
        destination = output / missing
        destination.mkdir(parents=True, exist_ok=False)
        for source in checkpoint.iterdir():
            if source.is_file() and source.name != missing:
                os.link(source, destination / source.name)
        try:
            load_policy(destination, device="cpu")
        except FileNotFoundError as error:
            if missing not in str(error):
                raise RuntimeError("checkpoint 文件检查没有定位实际缺少的文件") from error
        else:
            raise RuntimeError("checkpoint 缺少实际文件时需要终止加载")
        for command in ("play", "eval"):
            result = subprocess.run([sys.executable, "-m", "openso101.cli.main", "il", command,
                "--task", "OpenSO101-PickPlace-v0", "--policy-path", str(destination)],
                capture_output=True, text=True, timeout=30)
            if result.returncode == 0 or missing not in result.stderr or "AppLauncher" in result.stderr:
                raise RuntimeError("实际 IL 入口需要在启动 Isaac 前拒绝不完整 checkpoint")
        results.append({"missing_file": missing, "rejected": True, "native_startup_prevented": True})
    for path, device, error_type in (
        (output / "absent/model", "cpu", FileNotFoundError),
        (checkpoint / "config.json", "cpu", ValueError),
        ("invalid name/model", "cpu", ValueError),
        (checkpoint, "cuda:0", ValueError),
    ):
        try:
            load_policy(path, device=device)
        except error_type:
            pass
        else:
            raise RuntimeError("checkpoint 输入检查需要终止无效加载")
        results.append({"source": str(path), "device": device, "rejected": True})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("IL 训练准备验收需要禁止 CUDA")
    handle = load_lerobot_dataset(args.dataset)
    simulation, export_digest = dataset_simulation(handle.root, handle.dataset.meta.fps, handle.num_episodes)
    recording_path = handle.root / RECORDING_METADATA_FILE
    recording_digest = file_digest(recording_path) if recording_path.is_file() else None
    if args.output.exists() or args.output.resolve().is_relative_to(handle.root):
        raise ValueError("IL 检查输出需要使用数据集之外的新目录")
    source_files = sorted(path for folder in ("meta", "data", "videos")
                          for path in (handle.root / folder).rglob("*") if path.is_file())
    sources = {str(path.relative_to(handle.root)): file_digest(path) for path in source_files}
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for policy in ("act", "diffusion"):
        destination = args.output / policy
        training_output = args.output / f"training_{policy}"
        command = [sys.executable, "-m", "openso101.cli.main", "il", "train", "--policy", policy,
                   "--dataset", str(handle.root), "--output-dir", str(training_output), "--prepare-only",
                   "--preparation-output", str(destination), "--steps", "100000", "--batch-size", "8"]
        with (args.output / f"{policy}.log").open("x") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        report_path = destination / "report.json"
        report = json.loads(report_path.read_text())
        if (report["status"] != "actual_lerobot_training_preparation_verified" or report["training_started"]
                or report["gpu_tests_started"] or report["frames"] != handle.num_frames
                or not report["model_graph"]["model_forward_verified"]
                or not report["model_graph"]["model_backward_verified"]
                or not report["model_graph"]["model_inference_verified"]
                or not report["model_graph"]["model_checkpoint_verified"]
                or not report["model_graph"]["optimizer_configuration_verified"]
                or report["model_graph"]["optimizer_updates"] != 0 or training_output.exists()):
            raise RuntimeError("IL 实际模型和数据准备验收未通过")
        checkpoint = Path(report["model_graph"]["checkpoint_roundtrip"]["directory"])
        settings = load_simulation_settings(checkpoint)
        if (settings.recorded_simulation != simulation or settings.dataset_export_sha256 != export_digest
                or settings.dataset_recording_sha256 != recording_digest):
            raise RuntimeError("IL 完整模型没有保存实际导出或直接采集来源")
        rejected_checkpoints = check_checkpoint_inputs(checkpoint, args.output / f"checkpoint_inputs_{policy}")
        reports[policy] = {"report": str(report_path), "sha256": file_digest(report_path),
                           "model_graph": report["model_graph"],
                           "rejected_checkpoints": rejected_checkpoints,
                           "action_roundtrip_maximum_error": report["action_roundtrip_maximum_error"]}
    rejected = []
    cases = [
        ("act", ["--policy.chunk_size=0", "--policy.n_action_steps=0"], "序列长度必须为正整数"),
        ("act", ["--policy.dim_model=513"], "embed_dim must be divisible by num_heads"),
        ("diffusion", ["--policy.horizon=15"], "horizon should be an integer multiple"),
        ("diffusion", ["--policy.crop_shape=[4000,4000]"], "crop_shape"),
        ("act", ["--num_workers=-1"], "num_workers 必须为非负整数"),
        ("act", ["--save_checkpoint=false"], "需要保留模型和 optimizer checkpoint"),
        ("act", ["--policy.optimizer_lr=-1"], "optimizer lr 必须为有限正数"),
        ("diffusion", ["--policy.optimizer_lr=nan"], "optimizer lr 必须为有限正数"),
    ]
    for index, (policy, extras, message) in enumerate(cases):
        destination = args.output / f"rejected_{index}"
        training_output = args.output / f"rejected_training_{index}"
        result = subprocess.run([sys.executable, "-m", "openso101.cli.main", "il", "train",
            "--policy", policy, "--dataset", str(handle.root), "--output-dir", str(training_output),
            "--prepare-only", "--preparation-output", str(destination), "--", *extras],
            capture_output=True, text=True)
        (args.output / f"rejected_{index}.log").write_text(result.stdout + result.stderr)
        if result.returncode == 0 or message not in result.stderr or destination.exists() or training_output.exists():
            raise RuntimeError(f"IL 配置输入检查未拒绝实际命令: {extras}")
        rejected.append({"policy": policy, "arguments": extras, "exit_code": result.returncode})
    if {str(path.relative_to(handle.root)): file_digest(path) for path in source_files} != sources:
        raise RuntimeError("IL 验收期间的数据来源 SHA256 发生变化")
    report = {"status": "actual_il_training_preparation_verified", "policies": reports,
              "frames": handle.num_frames, "episodes": handle.num_episodes,
              "recorded_simulation_preserved": True,
              "dataset_recording_source_verified": recording_digest is not None,
              "dataset_recording_sha256": recording_digest,
              "rejected_configs": rejected, "source_files_unchanged": True,
              "source_files": sources, "gpu_tests_started": False, "training_started": False,
              "task_success_verified": False, "source_sha256": file_digest(Path(__file__))}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "frames", "episodes", "gpu_tests_started")}))


if __name__ == "__main__":
    main()
