# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .assets.catalog import AssetCatalog
from .models import Entity, SceneSpec, file_digest, scene_document_digest


def entity_bounds(entity: Entity) -> np.ndarray:
    half = np.asarray(entity.dimensions_m) / 2
    corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * half
    rotation = Rotation.from_quat(entity.pose.quaternion_wxyz, scalar_first=True)
    corners = rotation.apply(corners) + entity.pose.position
    return np.asarray([corners.min(axis=0), corners.max(axis=0)])


def validate_layout(spec: SceneSpec, catalog: AssetCatalog) -> dict:
    bounds = {}
    table = spec.table
    lower_xy = np.asarray(table.center_xy_m) - np.asarray(table.size_xy_m) / 2
    upper_xy = np.asarray(table.center_xy_m) + np.asarray(table.size_xy_m) / 2
    for entity in spec.entities:
        asset = catalog.read(entity.asset_uid)
        if asset.sha256 != entity.asset_sha256:
            raise ValueError(f"场景引用的资产版本不匹配：{entity.entity_id}")
        bound = entity_bounds(entity)
        reset_bound = bound + np.asarray(entity.reset_translation_m)
        if reset_bound[0, 2] < table.top_z_m - 1e-6:
            raise ValueError(f"物体或 reset 范围进入桌面：{entity.entity_id}")
        if (reset_bound[0, :2] < lower_xy).any() or (reset_bound[1, :2] > upper_xy).any():
            raise ValueError(f"物体或 reset 范围超出桌面：{entity.entity_id}")
        bounds[entity.entity_id] = bound.tolist()
    robot_xy = np.asarray(spec.robot_base.position[:2])
    if (robot_xy < lower_xy).any() or (robot_xy > upper_xy).any():
        raise ValueError("robot_base 超出桌面")
    goal = np.asarray(spec.task.goal_position_m)
    target = next(entity for entity in spec.entities if entity.entity_id == spec.task.object_id)
    target_bounds = entity_bounds(target) - np.asarray(target.pose.position) + goal
    if (target_bounds[0, :2] < lower_xy).any() or (target_bounds[1, :2] > upper_xy).any():
        raise ValueError("任务目标物体超出桌面")
    if target_bounds[0, 2] < table.top_z_m - 1e-6:
        raise ValueError("任务目标物体进入桌面")
    return {
        "scene_sha256": spec.digest(), "status": "layout_valid",
        "bounds_m": bounds,
        "pending_checks": ["collision", "stability", "reachability", "cameras", "collection"],
    }


def export_bundle(spec: SceneSpec, catalog: AssetCatalog, destination: Path, *, intent=None, program=None,
                  provenance: dict | None = None, bddl_report: dict | None = None) -> Path:
    if (intent is None) != (program is None):
        raise ValueError("SceneBundle 需要同时提供 TaskIntent 与 TaskProgram")
    if intent is not None:
        from .program import compile_program

        if compile_program(intent, spec) != program:
            raise ValueError("TaskProgram 必须保留 TaskIntent 的完整条件")
    report = validate_layout(spec, catalog)
    if bddl_report is not None:
        from .bddl import BDDLBinding, BDDLTaskTracker

        if bddl_report["scene_sha256"] != spec.digest():
            raise ValueError("BDDL 报告与场景版本不一致")
        BDDLTaskTracker(bddl_report["problem"], BDDLBinding.model_validate(bddl_report["binding"]), spec)
    destination = destination.resolve()
    # 已存在的 bundle 保持不可变，更新配置使用新的输出目录。
    destination.mkdir(parents=True, exist_ok=False)
    assets = AssetCatalog(destination / "assets")
    for uid in sorted({entity.asset_uid for entity in spec.entities}):
        shutil.copytree(catalog.directory(uid), assets.directory(uid))
    (destination / "scene.json").write_text(spec.model_dump_json(indent=2))
    if intent is not None:
        (destination / "task_intent.json").write_text(intent.model_dump_json(indent=2))
        (destination / "task_program.json").write_text(program.model_dump_json(indent=2))
    if provenance is not None:
        (destination / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    if bddl_report is not None:
        (destination / "bddl.json").write_text(json.dumps(bddl_report, ensure_ascii=False, indent=2))
    (destination / "validation.json").write_text(json.dumps(report, indent=2))
    files = {
        path.relative_to(destination).as_posix(): file_digest(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    (destination / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "scene_sha256": spec.digest(), "files": files,
    }, indent=2))
    return destination


def verify_bundle(root: Path) -> SceneSpec:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("bundle 缺少 manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("bundle manifest.json 无法读取") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("不支持的 bundle schema_version")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("bundle manifest 缺少 files 清单")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("bundle manifest files 必须是 path -> sha256 映射")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"bundle 文件不存在或路径越界：{relative}")
        if file_digest(path) != expected:
            raise ValueError(f"bundle 文件校验失败：{relative}")
    scene_path = root / "scene.json"
    try:
        spec = SceneSpec.read(scene_path)
    except (OSError, ValueError) as exc:
        raise ValueError("bundle scene.json 无法读取或格式无效") from exc
    # Older bundles were written with a canonical hash of the JSON document
    # (omitting Pydantic defaults), while current writers hash the validated
    # model.  Accept both representations so existing portable bundles stay
    # readable; all file hashes are still checked above.
    scene_hash = manifest.get("scene_sha256")
    if scene_hash not in {spec.digest(), scene_document_digest(scene_path)}:
        raise ValueError("scene_sha256 不匹配")
    catalog = AssetCatalog(root / "assets")
    validate_layout(spec, catalog)
    required = {"scene.json", "validation.json"}
    for entity in spec.entities:
        asset = catalog.read(entity.asset_uid)
        required.update(f"assets/{entity.asset_uid}/{name}" for name in (f"model.{asset.format}", "asset.json", "metadata.json"))
    if not required.issubset(files):
        raise ValueError("manifest 缺少必需文件")
    verify_program_documents(root, spec, files)
    return spec


def verify_program_documents(root: Path, spec: SceneSpec, files: dict):
    names = {"task_intent.json", "task_program.json"}
    if names.intersection(files) or any((root / name).exists() for name in names):
        if not names.issubset(files):
            raise ValueError("TaskIntent 与 TaskProgram 必须同时包含在文件校验清单中")
        from .program import TaskIntent, compile_program, read_program

        intent = TaskIntent.model_validate_json((root / "task_intent.json").read_text())
        if read_program(root / "task_program.json", spec) != compile_program(intent, spec):
            raise ValueError("TaskProgram 与 TaskIntent 的任务条件不一致")
    if (root / "bddl.json").exists() or "bddl.json" in files:
        from .bddl import BDDLBinding, BDDLTaskTracker

        if "bddl.json" not in files:
            raise ValueError("BDDL 文档必须包含在文件校验清单中")
        report = json.loads((root / "bddl.json").read_text())
        if report["scene_sha256"] != spec.digest():
            raise ValueError("BDDL 文档与场景 SHA256 不一致")
        BDDLTaskTracker(report["problem"], BDDLBinding.model_validate(report["binding"]), spec)
