# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

from .bundle import verify_bundle
from .catalog import AssetCatalog
from .models import SceneSpec, file_digest, scene_document_digest


async def convert_glb(source: Path, target: Path):
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.export_preview_surface = True
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.merge_all_meshes = True
    context.use_meter_as_world_unit = True
    context.baking_scales = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(target), None, context,
    )
    if not await task.wait_until_finished():
        raise RuntimeError(f"GLB 转换失败：{source}；{task.get_error_message()}")


def compose_stage(spec: SceneSpec, meshes: dict[str, Path], target: Path, *, include_physics_scene=True):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    if include_physics_scene:
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    table.AddTranslateOp().Set(Gf.Vec3d(*spec.table.center_xy_m, spec.table.top_z_m - spec.table.thickness_m / 2))
    table.AddScaleOp().Set(Gf.Vec3f(*spec.table.size_xy_m, spec.table.thickness_m))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Objects")
    for entity in spec.entities:
        root_path = f"/World/Objects/{entity.entity_id}"
        root = UsdGeom.Xform.Define(stage, root_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*entity.pose.position))
        quat = entity.pose.quaternion_wxyz
        root.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
        geometry = UsdGeom.Xform.Define(stage, root_path + "/Geometry")
        source = meshes[entity.asset_uid].resolve()
        source_stage = Usd.Stage.Open(str(source))
        if source_stage is None or not source_stage.GetDefaultPrim():
            raise ValueError(f"转换资产缺少 default prim：{source}")
        reference = stage.DefinePrim(root_path + "/Geometry/Source")
        reference.GetReferences().AddReference(Sdf.Reference(
            assetPath=Path(os.path.relpath(source, target.parent)).as_posix(),
        ))
        # 把资产坐标转换为 Z-up，并按用户填写的尺寸设置物体中心。
        rotation = Gf.Matrix4d(1.0)
        if UsdGeom.GetStageUpAxis(source_stage) == UsdGeom.Tokens.y:
            rotation.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
        geometry.AddTransformOp().Set(rotation)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bounds = cache.ComputeRelativeBound(geometry.GetPrim(), root.GetPrim()).ComputeAlignedRange()
        extent = bounds.GetSize()
        if any(value <= 0 for value in extent):
            raise ValueError(f"USD 资产尺寸无效：{entity.entity_id}")
        scale = Gf.Matrix4d(1).SetScale(Gf.Vec3d(*(size / extent[i] for i, size in enumerate(entity.dimensions_m))))
        center = Gf.Matrix4d(1).SetTranslate(-bounds.GetMidpoint())
        geometry.GetOrderedXformOps()[0].Set(rotation * center * scale)
        material = UsdShade.Material.Define(stage, root_path + "/PhysicsMaterial")
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr(entity.physics.static_friction)
        material_api.CreateDynamicFrictionAttr(entity.physics.dynamic_friction)
        material_api.CreateRestitutionAttr(entity.physics.restitution)
        mesh_count = 0
        for prim in Usd.PrimRange(geometry.GetPrim()):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(entity.physics.collision)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
                mesh_count += 1
        if mesh_count == 0:
            raise ValueError(f"USD 资产缺少 mesh：{entity.entity_id}")
        if entity.dynamic:
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(entity.physics.mass_kg)
    stage.GetRootLayer().Save()


def compile_bundle(bundle: Path, output: Path) -> Path:
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import UsdUtils

    spec = verify_bundle(bundle)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    enable_extension("omni.kit.asset_converter")
    meshes = {}
    catalog = AssetCatalog(bundle / "assets")
    for uid in sorted({entity.asset_uid for entity in spec.entities}):
        folder = output / "assets" / uid
        folder.mkdir(parents=True)
        asset = catalog.read(uid)
        if asset.format == "usdz":
            meshes[uid] = folder / "model.usdz"
            shutil.copyfile(catalog.directory(uid) / "model.usdz", meshes[uid])
        else:
            meshes[uid] = folder / "model.usd"
            asyncio.get_event_loop().run_until_complete(convert_glb(
                bundle.resolve() / "assets" / uid / "model.glb", meshes[uid],
            ))
        for name in ("asset.json", "metadata.json"):
            shutil.copyfile(bundle / "assets" / uid / name, folder / name)
    target = output / "scene.usda"
    compose_stage(spec, meshes, target)
    compose_stage(spec, meshes, output / "environment.usda", include_physics_scene=False)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(target))
    if unresolved:
        raise ValueError(f"USD 依赖无法加载：{unresolved}")
    dependencies = [layer.realPath for layer in layers] + list(assets)
    for dependency in dependencies:
        if not Path(dependency).resolve().is_relative_to(output):
            raise ValueError(f"USD 依赖位于输出目录以外：{dependency}")
    (output / "scene.json").write_text(spec.model_dump_json(indent=2))
    (output / "compilation.json").write_text(json.dumps({
        "scene_sha256": spec.digest(), "status": "compiled",
        "files": {path.relative_to(output).as_posix(): file_digest(path)
                  for path in sorted(output.rglob("*")) if path.is_file()},
        "pending_checks": ["simulation", "robot", "cameras", "collection"],
    }, indent=2))
    return target


def verify_compilation(output: Path) -> dict:
    output = output.resolve()
    report = json.loads((output / "compilation.json").read_text())
    if report["status"] != "compiled" or not {"scene.usda", "scene.json"}.issubset(report["files"]):
        raise ValueError("编译报告缺少必需内容")
    for relative, expected in report["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or file_digest(path) != expected:
            raise ValueError(f"编译文件校验失败：{relative}")
    SceneSpec.read(output / "scene.json")
    if scene_document_digest(output / "scene.json") != report["scene_sha256"]:
        raise ValueError("编译配置的 SHA256 不匹配")
    return report
