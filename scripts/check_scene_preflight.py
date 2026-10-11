import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from openso101.scenes.bundle import verify_bundle
from openso101.scenes.isaaclab.preflight import validate_runtime_scene, validate_scene_output
from openso101.scenes.isaaclab.preparation import prepare_scene
from openso101.scenes.models import file_digest


def reject_request(name, command, expected, directory):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    stdout = directory / f"{name}.stdout.txt"
    stderr = directory / f"{name}.stderr.txt"
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    if result.returncode == 0 or expected not in result.stderr:
        raise RuntimeError(f"场景启动请求检查失败：{name}")
    if "Simulation App Starting" in result.stdout + result.stderr:
        raise RuntimeError(f"场景启动检查启动了原生应用：{name}")
    return {"name": name, "command": command, "exit_code": result.returncode,
            "expected_error": expected, "stdout_sha256": file_digest(stdout),
            "stderr_sha256": file_digest(stderr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OPENSO101_SKIP_ISAAC") != "1":
        raise ValueError("场景启动检查需要禁止使用 CUDA 与 Isaac")
    args.bundle = args.bundle.resolve()
    args.scene = args.scene.resolve()
    args.episode = args.episode.resolve()
    spec = verify_bundle(args.bundle)
    compilation = validate_runtime_scene(args.scene)
    if compilation["scene_sha256"] != spec.digest():
        raise ValueError("启动检查的编译场景与源 bundle 不匹配")
    args.output = validate_scene_output(args.bundle, args.output)
    validate_scene_output(args.scene, args.output)
    if args.episode.is_relative_to(args.output):
        raise ValueError("来源 episode 需要位于检查输出目录以外")
    sources = {path: file_digest(path) for directory in (args.bundle, args.scene)
               for path in directory.rglob("*") if path.is_file()}
    sources[args.episode] = file_digest(args.episode)
    args.output.mkdir(parents=True, exist_ok=False)
    work = args.output / "requests"
    work.mkdir()
    api_records = []
    for name in ("num_envs", "steps", "resets"):
        for index, value in enumerate((True, False, 0, -1, 1.5, "4", None)):
            output = work / f"invalid_{name}_{index}"
            options = {"num_envs": 4, "steps": 200, "resets": 100, name: value}
            try:
                prepare_scene(args.bundle, output, **options)
            except ValueError as error:
                if name not in str(error):
                    raise
            else:
                raise RuntimeError(f"无效场景参数需要终止：{name}")
            if output.exists():
                raise RuntimeError(f"无效场景参数创建了输出目录：{name}")
            api_records.append({"field": name, "value": value, "output_created": False})
    for value in (True, False, 0, -1, float("inf"), float("nan"), "1"):
        output = work / f"invalid_timeout_{len(api_records)}"
        try:
            prepare_scene(args.bundle, output, timeout_seconds=value)
        except ValueError as error:
            if "timeout_seconds" not in str(error):
                raise
        else:
            raise RuntimeError("无效场景预算需要终止")
        if output.exists():
            raise RuntimeError("无效场景预算创建了输出目录")
        api_records.append({"field": "timeout_seconds", "value": str(value), "output_created": False})

    copied = work / "compiled_without_environment_manifest"
    shutil.copytree(args.scene, copied)
    report_path = copied / "compilation.json"
    modified = json.loads(report_path.read_text())
    del modified["files"]["environment.usda"]
    report_path.write_text(json.dumps(modified, indent=2))
    trace = work / "existing_trace.hdf5"
    shutil.copyfile(args.episode, trace)
    trace_sha256 = file_digest(trace)
    worker = [sys.executable, "-u", "-m", "openso101.scenes.isaaclab.worker"]
    runtime = [sys.executable, "-u", "-m", "openso101.scenes.isaaclab.validation_worker"]
    command = [sys.executable, "-u", "-m", "openso101.cli.main"]
    cli = [*command, "scenes"]
    training = [*command, "rl", "train", "--backend", "rsl_rl", "--algo", "ppo"]
    cases = [
        ("compile_missing_bundle", [*worker, str(work / "absent"), str(work / "missing_bundle_output")],
         "bundle 缺少 manifest.json"),
        ("compile_output_inside_source", [*worker, str(args.bundle), str(args.bundle / "forbidden_output")],
         "场景输出目录与输入目录必须独立"),
        ("compile_existing_output", [*worker, str(args.bundle), str(copied)], "FileExistsError"),
        ("compile_cuda_disabled", [*worker, str(args.bundle), str(work / "compile_cuda_output")],
         "请求的 GPU [] 超出允许范围"),
        ("runtime_missing_scene", [*runtime, str(work / "absent"), str(work / "missing_scene.json")],
         "FileNotFoundError"),
        ("runtime_missing_environment_manifest", [*runtime, str(copied), str(work / "manifest.json")],
         "原生场景需要通过 SHA256 检查的 environment.usda"),
        ("runtime_existing_trace", [*runtime, str(args.scene), str(trace.with_suffix(".json"))],
         "FileExistsError"),
        ("runtime_invalid_steps", [*runtime, str(args.scene), str(work / "steps.json"), "--steps", "0"],
         "steps 必须为正整数"),
        ("runtime_cuda_disabled", [*runtime, str(args.scene), str(work / "cuda_disabled.json")],
         "请求的 GPU [] 超出允许范围"),
        ("cli_prepare_invalid_resets", [*cli, "prepare", str(args.bundle), "--output", str(work / "prepare"),
                                        "--resets", "0"], "resets 必须为正整数"),
        ("cli_compile_output_inside_source", [*cli, "compile", str(args.bundle), "--output",
                                               str(args.bundle / "forbidden_output")],
         "场景输出目录与输入目录必须独立"),
        ("cli_runtime_existing_trace", [*cli, "validate-runtime", str(args.scene), "--output",
                                         str(trace.with_suffix(".json"))], "FileExistsError"),
        ("cli_envs_missing_environment_manifest", [*command, "envs", "zero", "--task", "OpenSO101-CustomScene-v0",
                                                   "--scene", str(copied), "--headless"],
         "原生场景需要通过 SHA256 检查的 environment.usda"),
        ("cli_il_missing_environment_manifest", [*command, "il", "record", "--task", "OpenSO101-CustomScene-v0",
                                                 "--scene", str(copied), "--repo-root", str(work / "il_record"),
                                                 "--teleop-device", "keyboard", "--keyboard-input", "window"],
         "原生场景需要通过 SHA256 检查的 environment.usda"),
        ("cli_rl_missing_environment_manifest", [*training, "--task", "OpenSO101-CustomScene-v0",
                                                 "--scene", str(copied), "--output", str(work / "rl_missing_manifest")],
         "原生场景需要通过 SHA256 检查的 environment.usda"),
        ("cli_rl_output_inside_source", [*training, "--task", "OpenSO101-CustomScene-v0", "--scene", str(args.scene),
                                         "--output", str(args.scene / "forbidden_output")],
         "场景输出目录与输入目录必须独立"),
        ("cli_rl_task_scene_mismatch", [*training, "--task", "OpenSO101-Lift-v0", "--scene", str(args.scene),
                                        "--output", str(work / "rl_task_mismatch")], "CustomScene 请求需要对应场景"),
        ("cli_rl_missing_scene", [*training, "--task", "OpenSO101-CustomScene-v0",
                                  "--output", str(work / "rl_missing_scene")], "CustomScene 请求需要对应场景"),
    ]
    records = [reject_request(name, command, expected, args.output) for name, command, expected in cases]
    if set(work.iterdir()) != {copied, trace}:
        raise RuntimeError("终止的场景请求创建了输出文件")
    if file_digest(trace) != trace_sha256 or any(file_digest(path) != digest for path, digest in sources.items()):
        raise RuntimeError("场景启动检查改变了实际输入文件")
    report = {
        "status": "actual_scene_preflight_verified", "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "bundle": str(args.bundle), "scene_sha256": spec.digest(),
        "bundle_manifest_sha256": file_digest(args.bundle / "manifest.json"),
        "compiled_scene": str(args.scene), "compiled_scene_sha256": compilation["scene_sha256"],
        "compilation_sha256": file_digest(args.scene / "compilation.json"),
        "source_file_count": len(sources), "source_files_unchanged": True,
        "episode_sha256": file_digest(args.episode), "existing_trace_preserved": True,
        "api_count": len(api_records), "api_checks": api_records,
        "process_count": len(records), "process_checks": records,
        "source_sha256": file_digest(Path(__file__)),
        "gpu_tests_started": False, "native_tests_started": False, "training_started": False,
        "physical_task_verified": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "api_count": report["api_count"],
                      "process_count": report["process_count"], "gpu_tests_started": False}))


if __name__ == "__main__":
    main()
