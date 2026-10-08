import argparse
import json
import os
from pathlib import Path

from pydantic import Field, model_validator

from openso101.scenes.bundle import verify_bundle
from openso101.scenes.assets.catalog import AssetCatalog
from openso101.scenes.layout import diagnose_layout
from openso101.scenes.models import Identifier, Model, file_digest


class SceneCase(Model):
    name: Identifier
    bundle: str


class SceneCases(Model):
    cases: tuple[SceneCase, ...] = Field(min_length=10)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({item.name for item in self.cases}) != len(self.cases):
            raise ValueError("场景验收需要唯一的名称")
        return self


parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--phase", choices=("cpu", "gpu"), required=True)
parser.add_argument("--cpu-report", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cases = SceneCases.model_validate_json(args.cases.read_text())
repo = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True, exist_ok=False)
records = []
for case in cases.cases:
    bundle = (repo / case.bundle).resolve()
    if not bundle.is_relative_to(repo / "outputs"):
        raise ValueError("场景验收需要仓库 outputs 中的实际 bundle")
    spec = verify_bundle(bundle)
    diagnostics = diagnose_layout(spec, AssetCatalog(bundle / "assets"))
    record = {"name": case.name, "bundle": case.bundle, "scene_sha256": spec.digest(),
              "manifest_sha256": file_digest(bundle / "manifest.json"), "diagnostics": diagnostics}
    records.append(record)
    if diagnostics["status"] != "static_checks_passed":
        (args.output / "rejected_case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        raise ValueError(f"实际场景的静态检查未通过：{case.name}")
report = {"status": "scene_batch_cpu_verified", "cases_sha256": file_digest(args.cases),
          "cases": records, "count": len(records), "source_sha256": file_digest(Path(__file__)),
          "gpu_tests_started": False, "task_success_verified": False, "dataset_verified": False}
if args.phase == "gpu":
    if args.cpu_report is None:
        raise ValueError("GPU 场景验收需要本次 CPU 报告")
    previous = json.loads(args.cpu_report.read_text())
    if previous != report:
        raise ValueError("GPU 场景验收的输入与通过 CPU 检查的记录不一致")
    from openso101.scenes.isaaclab.preparation import prepare_scene

    report.update(status="scene_batch_gpu_running", gpu_tests_started=True, runtime_results=[])
    for case in cases.cases:
        result = prepare_scene(repo / case.bundle, args.output / case.name, num_envs=4, steps=200, resets=100,
                               timeout_seconds=1800)
        report["runtime_results"].append({"name": case.name, "runtime_report_sha256": result["runtime_report_sha256"],
                                         "preparation_sha256": file_digest(args.output / case.name / "preparation.json")})
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report["status"] = "scene_batch_runtime_verified"
elif os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("CPU 场景检查需要禁止使用 CUDA")
(args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": report["status"], "count": report["count"], "task_success_verified": False}))
