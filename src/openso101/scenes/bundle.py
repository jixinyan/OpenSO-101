# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .catalog import AssetCatalog
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


def export_bundle(spec: SceneSpec, catalog: AssetCatalog, destination: Path) -> Path:
    report = validate_layout(spec, catalog)
    destination = destination.resolve()
    # 已存在的 bundle 保持不可变，更新配置使用新的输出目录。
    destination.mkdir(parents=True, exist_ok=False)
    assets = AssetCatalog(destination / "assets")
    for uid in sorted({entity.asset_uid for entity in spec.entities}):
        shutil.copytree(catalog.directory(uid), assets.directory(uid))
    (destination / "scene.json").write_text(spec.model_dump_json(indent=2))
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
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("不支持的 bundle schema_version")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_digest(path) != expected:
            raise ValueError(f"bundle 文件校验失败：{relative}")
    spec = SceneSpec.read(root / "scene.json")
    if scene_document_digest(root / "scene.json") != manifest["scene_sha256"]:
        raise ValueError("scene_sha256 不匹配")
    catalog = AssetCatalog(root / "assets")
    validate_layout(spec, catalog)
    required = {"scene.json", "validation.json"}
    for entity in spec.entities:
        asset = catalog.read(entity.asset_uid)
        required.update(f"assets/{entity.asset_uid}/{name}" for name in (f"model.{asset.format}", "asset.json", "metadata.json"))
    if not required.issubset(manifest["files"]):
        raise ValueError("manifest 缺少必需文件")
    return spec
