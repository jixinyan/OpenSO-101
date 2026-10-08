# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import objaverse
import trimesh

from ..models import Asset, file_digest


class AssetCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, uid: str) -> Path:
        # 通过字段校验限制路径中的 identifier。
        if len(uid) != 32 or any(char not in "0123456789abcdef" for char in uid):
            raise ValueError("Objaverse UID 必须为 32 位小写十六进制字符串")
        return self.root / uid

    def read(self, uid: str) -> Asset:
        folder = self.directory(uid)
        asset = Asset.model_validate_json((folder / "asset.json").read_text())
        if asset.uid != uid or file_digest(folder / f"model.{asset.format}") != asset.sha256:
            raise ValueError(f"资产内容与记录不匹配：{uid}")
        return asset

    def list(self) -> list[Asset]:
        return [self.read(path.parent.name) for path in sorted(self.root.glob("*/asset.json"))]

    def import_glb(self, path: Path, *, uid: str, metadata: dict) -> Asset:
        folder = self.directory(uid)
        # glTF 使用 Y-up；场景实例通过显式旋转转换为 Z-up。
        mesh = trimesh.load(path, file_type="glb", force="mesh", process=False)
        if not np.isfinite(mesh.vertices).all() or (mesh.extents <= 0).any():
            raise ValueError("资产必须包含三维、有限尺寸的 mesh")
        asset = Asset(
            uid=uid, name=metadata["name"], source_url=metadata["viewerUrl"],
            license=metadata["license"], author=metadata["user"]["displayName"],
            sha256=file_digest(path), bounds=mesh.bounds.tolist(),
            vertices=len(mesh.vertices), faces=len(mesh.faces),
        )
        if folder.exists():
            existing = self.read(uid)
            if existing != asset:
                raise FileExistsError(f"同一 UID 已有其他内容：{uid}")
            return existing
        folder.mkdir()
        shutil.copyfile(path, folder / "model.glb")
        (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        (folder / "asset.json").write_text(asset.model_dump_json(indent=2))
        return asset

    def fetch(self, uid: str) -> Asset:
        folder = self.directory(uid)
        if folder.exists():
            return self.read(uid)
        annotations = objaverse.load_annotations([uid])
        metadata = annotations[uid]
        if not metadata.get("license"):
            raise ValueError(f"资产缺少 license：{uid}")
        objects = objaverse.load_objects([uid], download_processes=1)
        return self.import_glb(Path(objects[uid]), uid=uid, metadata=metadata)


def search_categories(query: str, limit: int = 20) -> list[dict]:
    if not query.strip() or not 1 <= limit <= 100:
        raise ValueError("query 不能为空，limit 必须位于 1 到 100")
    categories = objaverse.load_lvis_annotations()
    return [
        {"category": category, "count": len(uids), "uids": uids[:limit]}
        for category, uids in sorted(categories.items())
        if query.casefold() in category.casefold()
    ][:limit]
