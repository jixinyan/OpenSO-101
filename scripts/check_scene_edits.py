import argparse
import json
import os
from pathlib import Path

import numpy as np

from openso101.scenes.bundle import verify_bundle
from openso101.scenes.editor.editing import interpret_scene_edit, revise_bundle
from openso101.scenes.agent.model_client import ModelService, load_codex_runtime_config
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("bundle", type=Path)
parser.add_argument("--model-config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
config = load_codex_runtime_config(args.model_config)
key_environment = "OPENSO101_EDIT_MODEL_KEY"
if config.bearer_token:
    os.environ[key_environment] = config.bearer_token
args.output.mkdir(parents=True)
bundle = args.bundle.resolve()
records = []
for axis, delta in ((0, -.02), (1, .02), (0, .02)):
    before = verify_bundle(bundle)
    entity = before.entities[0]
    labels = ("x", "y", "z")
    instruction = (f"仅将实体 {entity.entity_id} 的初始位置沿 {labels[axis]} 轴移动 {delta} 米。"
                   "保持另外两个坐标、旋转、尺寸、任务目标、任务文本和全部其他配置。")
    service = ModelService(config.base_url, config.model, key_environment, wire_api=config.wire_api,
                           reasoning_effort="low", max_requests=1, max_retries=0)
    edit = interpret_scene_edit(bundle, instruction, service)
    target = args.output / f"revision_{len(records) + 1}"
    report = revise_bundle(bundle, edit, target, model_requests=service.requests)
    after = verify_bundle(target)
    expected = np.asarray(entity.pose.position).copy()
    expected[axis] += delta
    measured = np.asarray(after.entities[0].pose.position)
    if not np.allclose(measured, expected, rtol=0, atol=1e-8):
        raise RuntimeError("实际模型修改的位移与用户要求不一致")
    unchanged_before = before.model_dump()
    unchanged_after = after.model_dump()
    unchanged_before["entities"][0]["pose"]["position"] = tuple(measured)
    if unchanged_before != unchanged_after:
        raise RuntimeError("实际模型改变了用户未要求修改的配置")
    records.append({"instruction": instruction, "edit": edit.model_dump(mode="json"),
                    "requests": service.requests, "result": report,
                    "maximum_position_error_m": float(np.max(np.abs(measured - expected))),
                    "unchanged_conditions_verified": True})
    bundle = target
report = {"status": "actual_model_scene_edits_verified", "model": config.model,
          "revisions": records, "source_bundle_manifest_sha256": file_digest(args.bundle / "manifest.json"),
          "source_sha256": file_digest(Path(__file__)), "gpu_runtime_verified": False,
          "task_success_verified": False}
with (args.output / "report.json").open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps({"status": report["status"], "model": config.model, "revisions": len(records)}, ensure_ascii=False))
