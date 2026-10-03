import json
from pathlib import Path
import subprocess
import sys

from .bundle import verify_bundle
from .models import file_digest
from .usd import verify_compilation


def prepare_scene(bundle: Path, output: Path, *, num_envs: int = 4, steps: int = 200, resets: int = 100,
                  timeout_seconds: float | None = None) -> dict:
    if min(num_envs, steps, resets) <= 0:
        raise ValueError("环境数量、步骤数量和 reset 次数必须大于零")
    bundle = bundle.resolve()
    output = output.resolve()
    spec = verify_bundle(bundle)
    if output.is_relative_to(bundle):
        raise ValueError("场景准备输出需要位于源 bundle 目录以外")
    output.mkdir(parents=True, exist_ok=False)
    compiled = output / "compiled"
    runtime_path = output / "runtime.json"
    import math
    import time

    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("场景运行预算必须为正数有限数值")
    started = time.monotonic()
    subprocess.run([sys.executable, "-u", "-m", "openso101.scenes.worker", str(bundle), str(compiled)],
                   check=True, timeout=timeout_seconds)
    compilation = verify_compilation(compiled)
    remaining = None if timeout_seconds is None else timeout_seconds - (time.monotonic() - started)
    if remaining is not None and remaining <= 0:
        raise TimeoutError("场景编译已达到运行预算上限")
    subprocess.run([
        sys.executable, "-u", "-m", "openso101.scenes.validation_worker", str(compiled), str(runtime_path),
        "--num-envs", str(num_envs), "--steps", str(steps), "--resets", str(resets), "--cameras",
    ], check=True, timeout=remaining)
    runtime = json.loads(runtime_path.read_text())
    if compilation["scene_sha256"] != spec.digest() or runtime["scene_sha256"] != spec.digest():
        raise ValueError("场景准备报告与源场景不匹配")
    if runtime["status"] != "runtime_verified" or (runtime["num_envs"], runtime["steps"], runtime["resets"]) != (num_envs, steps, resets):
        raise ValueError("场景运行检查数量或状态不匹配")
    for name in ("wrist_camera", "overhead_camera"):
        if runtime["camera_checks"][name]["checked_frames"] != num_envs * steps:
            raise ValueError(f"相机检查数量不匹配: {name}")
    report = {
        "status": "runtime_verified", "source_bundle": str(bundle), "scene_sha256": spec.digest(),
        "source_manifest_sha256": file_digest(bundle / "manifest.json"),
        "compiled_scene": str(compiled), "compilation_sha256": file_digest(compiled / "compilation.json"),
        "runtime_report_sha256": file_digest(runtime_path), "runtime": runtime,
        "task_success_verified": False, "dataset_verified": False,
        "pending_checks": runtime["pending_checks"], "preparer_sha256": file_digest(Path(__file__)),
    }
    with (output / "preparation.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    return report
