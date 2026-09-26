# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from .catalog import AssetCatalog
from .models import Asset, file_digest


def import_robotwin(catalog: AssetCatalog, model_directory: Path, model_id: int, *, license: str, author: str) -> Asset:
    if model_id < 0:
        raise ValueError("model_id 不能为负数")
    source = model_directory.resolve() / "visual" / f"base{model_id}.glb"
    metadata_path = model_directory / f"model_data{model_id}.json"
    source_metadata = json.loads(metadata_path.read_text())
    scale = np.asarray(source_metadata["scale"])
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("RoboTwin 资产 scale 无效")
    metadata = {
        "name": f"{model_directory.name}_{model_id}", "viewerUrl": source.as_uri(),
        "license": license, "user": {"displayName": author}, "provider": "robotwin",
        "source_metadata": source_metadata, "source_metadata_sha256": file_digest(metadata_path),
    }
    return catalog.import_glb(source, uid=file_digest(source)[:32], metadata=metadata)


def import_usd(catalog: AssetCatalog, source: Path, prim_path: str, *, name: str, license: str, author: str) -> Asset:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    source = source.resolve()
    original = Usd.Stage.Open(str(source))
    prim = original.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"USD 中不存在 prim：{prim_path}")
    if any(child.IsA(UsdPhysics.Joint) or child.HasAPI(UsdPhysics.ArticulationRootAPI) for child in Usd.PrimRange(prim)):
        raise ValueError("该资产包含 articulation，需要 articulation task 类型")
    scratch = catalog.root / ".imports" / uuid.uuid4().hex
    scratch.mkdir(parents=True)
    selected = Usd.Stage.CreateNew(str(scratch / "selected.usda"))
    root = selected.DefinePrim("/Asset")
    root.GetReferences().AddReference(str(source), prim_path)
    selected.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(selected, UsdGeom.GetStageUpAxis(original))
    UsdGeom.SetStageMetersPerUnit(selected, UsdGeom.GetStageMetersPerUnit(original))
    selected.Flatten().Export(str(scratch / "geometry.usda"))
    geometry = Usd.Stage.Open(str(scratch / "geometry.usda"))
    vertices = faces = 0
    for child in geometry.Traverse():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if child.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(child)
            vertices += len(mesh.GetPointsAttr().Get())
            faces += len(mesh.GetFaceVertexCountsAttr().Get())
    if vertices == 0 or faces == 0:
        raise ValueError("USD 资产必须包含 mesh")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = cache.ComputeWorldBound(geometry.GetDefaultPrim()).ComputeAlignedRange()
    if any(size <= 0 for size in bounds.GetSize()):
        raise ValueError("USD 资产尺寸必须为正数")
    geometry.GetRootLayer().Save()
    package = scratch / "model.usdz"
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(scratch / "geometry.usda")), str(package)):
        raise RuntimeError("USDZ 打包失败")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(package))
    if unresolved:
        raise ValueError(f"USD 资产缺少依赖：{unresolved}")
    sha256 = file_digest(package)
    metadata = {"name": name, "viewerUrl": source.as_uri(), "license": license,
                "user": {"displayName": author}, "provider": "usd", "prim_path": prim_path,
                "source_sha256": file_digest(source)}
    asset = Asset(uid=sha256[:32], name=name, source_url=source.as_uri(), license=license,
                  author=author, sha256=sha256, bounds=(tuple(bounds.GetMin()), tuple(bounds.GetMax())),
                  vertices=vertices, faces=faces, format="usdz")
    folder = catalog.directory(asset.uid)
    if folder.exists():
        return catalog.read(asset.uid)
    folder.mkdir()
    shutil.copyfile(package, folder / "model.usdz")
    (folder / "asset.json").write_text(asset.model_dump_json(indent=2))
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return asset
