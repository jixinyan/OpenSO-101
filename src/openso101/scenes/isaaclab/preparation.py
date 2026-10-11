import json
from pathlib import Path
import subprocess
import sys
import time

from ..bundle import verify_bundle
from ..models import file_digest
from .preflight import validate_runtime_parameters, validate_runtime_scene, validate_scene_output


def prepare_scene(bundle: Path, output: Path, *, num_envs: int = 4, steps: int = 200, resets: int = 100,
                  timeout_seconds: float | None = None) -> dict:
    validate_runtime_parameters(num_envs=num_envs, steps=steps, resets=resets,
                                timeout_seconds=timeout_seconds)
    bundle = bundle.resolve()
    output = validate_scene_output(bundle, output)
    spec = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    compiled = output / "compiled"
    runtime_path = output / "runtime.json"
    started = time.monotonic()
    subprocess.run([sys.executable, "-u", "-m", "openso101.scenes.isaaclab.worker", str(bundle), str(compiled)],
                   check=True, timeout=timeout_seconds)
    compilation = validate_runtime_scene(compiled)
    remaining = None if timeout_seconds is None else timeout_seconds - (time.monotonic() - started)
    if remaining is not None and remaining <= 0:
        raise TimeoutError("场景编译已达到运行预算上限")
    subprocess.run([
        sys.executable, "-u", "-m", "openso101.scenes.isaaclab.validation_worker", str(compiled), str(runtime_path),
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
