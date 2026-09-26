# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import numpy as np
import trimesh

from .catalog import AssetCatalog


def inspect_asset(catalog: AssetCatalog, uid: str) -> dict:
    asset = catalog.read(uid)
    if asset.format == "glb":
        geometry = trimesh.load(catalog.directory(uid) / "model.glb", force="mesh", process=False)
        report = {
            "source_up_axis": "Y", "source_dimensions": geometry.extents.tolist(),
            "source_units": "unspecified", "watertight": bool(geometry.is_watertight),
            "winding_consistent": bool(geometry.is_winding_consistent),
            "connected_components": len(geometry.split(only_watertight=False)),
            "finite_vertices": bool(np.isfinite(geometry.vertices).all()),
        }
    else:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(catalog.directory(uid) / "model.usdz"))
        report = {"source_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                  "source_dimensions": np.diff(asset.bounds, axis=0)[0].tolist(),
                  "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                  "vertices": asset.vertices, "faces": asset.faces}
    return {
        "asset_uid": uid,
        "asset_sha256": asset.sha256,
        "source": asset.source_url,
        "license": asset.license,
        "geometry": report,
        "capabilities": {name: {"status": "unknown", "evidence": []} for name in (
            "graspable", "support_surface", "container_interior", "articulated", "pourable",
        )},
        "required_instance_properties": ["dimensions_m", "physics", "robot"],
    }
