# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import shutil

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from openso101.scenes.catalog import AssetCatalog
from openso101.scenes.importers import import_usd
from openso101.scenes.models import Entity, Pose, SceneSpec, Task, file_digest
from openso101.scenes.usd import compose_stage


def test_compose_real_usd_geometry(tmp_path):
    source = tmp_path / "object.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/Object")
    root.AddTranslateOp().Set(Gf.Vec3d(7, 2, -3))
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Object/Tetrahedron")
    mesh.CreatePointsAttr([(0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    stage.GetRootLayer().Save()
    uid = file_digest(source)[:32]
    spec = SceneSpec(scene_id="usd_test", entities=(Entity(
        entity_id="object", asset_uid=uid, asset_sha256=file_digest(source),
        dimensions_m=(0.04, 0.06, 0.08), pose=Pose(position=(0.25, 0, 0.04)),
    ),), task=Task(task_id="place", object_id="object", goal_position_m=(0.25, 0.15, 0.04)))
    output = tmp_path / "scene.usda"
    compose_stage(spec, {uid: source}, output)
    scene = Usd.Stage.Open(str(output))
    prim = scene.GetPrimAtPath("/World/Objects/object")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    np.testing.assert_allclose(bounds.GetSize(), (0.04, 0.06, 0.08), atol=1e-7)
    np.testing.assert_allclose(bounds.GetMidpoint(), (0.25, 0, 0.04), atol=1e-7)
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0
    assert scene.GetPrimAtPath("/World/Table").HasAPI(UsdPhysics.CollisionAPI)
    assert scene.GetPrimAtPath("/World/Objects/object/Geometry/Source/Tetrahedron").HasAPI(UsdPhysics.CollisionAPI)
    moved = tmp_path / "portable"
    moved.mkdir()
    shutil.copy(source, moved / source.name)
    shutil.copy(output, moved / output.name)
    portable = Usd.Stage.Open(str(moved / output.name))
    assert portable.GetPrimAtPath("/World/Objects/object/Geometry/Source/Tetrahedron")


def test_import_selected_usd_prim_is_portable(tmp_path):
    source = tmp_path / "external.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/Room")
    UsdGeom.Cube.Define(stage, "/Room/Unselected")
    mesh = UsdGeom.Mesh.Define(stage, "/Room/Object")
    mesh.CreatePointsAttr([(0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    stage.GetRootLayer().Save()
    catalog = AssetCatalog(tmp_path / "assets")
    asset = import_usd(catalog, source, "/Room/Object", name="Tetrahedron", license="CC0", author="OpenSO-101 tests")
    assert asset.format == "usdz"
    assert catalog.read(asset.uid) == asset
    portable = Usd.Stage.Open(str(catalog.directory(asset.uid) / "model.usdz"))
    assert portable.GetDefaultPrim().IsA(UsdGeom.Mesh)
    assert not portable.GetPrimAtPath("/Room/Unselected")
    np.testing.assert_allclose(np.diff(asset.bounds, axis=0)[0], (2, 3, 4))
