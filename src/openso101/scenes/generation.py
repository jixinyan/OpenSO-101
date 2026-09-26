# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

from .bundle import validate_layout
from .catalog import AssetCatalog
from .model_client import ModelService, schema_instruction
from .models import SceneSpec


def propose_scene(instruction: str, catalog: AssetCatalog, service: ModelService) -> SceneSpec:
    if not instruction.strip():
        raise ValueError("任务描述不能为空")
    assets = catalog.list()
    if not assets:
        raise ValueError("资产目录为空，请导入任务需要的资产")
    system = (
        "根据用户任务和资产目录生成 SO-101 桌面操作场景，仅返回符合 schema 的 JSON。"
        "所有长度使用米，Z 轴向上，quaternion 使用 wxyz。"
        "仅能使用目录中的 UID 和 sha256，保存完整用户指令到 task.instruction。"
        "物体必须位于桌面上方；dimensions_m 为旋转之前的尺寸。"
        "为 SO-101 选择合理尺寸和操作距离，明确记录所有任务目标。"
        "容器内部区域需要用户提供的测量结果；不能凭名称编造容器内部区域。"
        "质量和摩擦等估计值使用 estimated provenance。"
        "schema：" + schema_instruction(SceneSpec)
    )
    prompt = json.dumps({"instruction": instruction, "assets": [asset.model_dump() for asset in assets]}, ensure_ascii=False)
    spec = service.complete(system=system, prompt=prompt, schema=SceneSpec)
    if spec.task.instruction != instruction:
        raise ValueError("模型必须完整保留用户任务描述")
    validate_layout(spec, catalog)
    return spec


def save_proposal(spec: SceneSpec, output: Path, service: ModelService) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    (output / "scene.json").write_text(spec.model_dump_json(indent=2))
    (output / "generation.json").write_text(json.dumps({
        "model": service.model,
        "scene_sha256": spec.digest(),
        "status": "layout_valid",
        "pending_checks": ["physics", "reachability", "camera_visibility", "task_execution"],
    }, ensure_ascii=False, indent=2))
    return output / "scene.json"
