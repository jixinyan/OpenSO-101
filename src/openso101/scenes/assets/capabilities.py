from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from pydantic import Field, model_validator

from .catalog import AssetCatalog
from ..models import Digest, Dimensions, Model, Vector3, file_digest


class GeometryProbe(Model):
    dimensions_m: Dimensions
    capability: Literal["container", "support_surface", "graspable"]
    region_bounds_m: tuple[Vector3, Vector3] | None = None
    maximum_gripper_opening_m: float | None = Field(default=None, gt=0)
    robot_sha256: Digest | None = None
    collision: Literal["convexHull", "convexDecomposition"]
    samples_per_axis: int = Field(default=9, ge=3, le=31)
    clearance_m: float = Field(default=0.001, gt=0)

    @model_validator(mode="after")
    def required_measurements(self):
        if self.capability in ("container", "support_surface") and self.region_bounds_m is None:
            raise ValueError("容器与支撑面检查需要指定实际检查区域")
        if self.region_bounds_m is not None:
            lower, upper = self.region_bounds_m
            if any(a >= b for a, b in zip(lower, upper)):
                raise ValueError("检查区域必须具有正尺寸")
        if self.capability == "graspable" and (
            self.maximum_gripper_opening_m is None or self.robot_sha256 is None
        ):
            raise ValueError("夹持几何检查需要机器人文件 SHA256 和实际夹爪开口")
        return self


def instance_mesh(catalog: AssetCatalog, uid: str, dimensions_m: Dimensions, *, require_volume=False) -> trimesh.Trimesh:
    asset = catalog.read(uid)
    if asset.format != "glb":
        raise ValueError("几何能力检查需要 GLB；USD 资产需要独立的 mesh 导出")
    mesh = trimesh.load(catalog.directory(uid) / "model.glb", force="mesh", process=True)
    if not np.isfinite(mesh.vertices).all() or (mesh.extents <= 0).any():
        raise ValueError("资产 mesh 必须具有有限坐标和正尺寸")
    transform = trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0))
    mesh.apply_transform(transform)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(np.asarray(dimensions_m) / mesh.extents)
    if require_volume and (not mesh.is_watertight or not mesh.is_winding_consistent or not mesh.is_volume):
        raise ValueError("体积与内部区域检查需要封闭、朝向一致且具有正体积的实际 mesh")
    return mesh


def _grid(bounds, count):
    lower, upper = np.asarray(bounds)
    return np.stack(np.meshgrid(*(np.linspace(a, b, count) for a, b in zip(lower, upper)),
                               indexing="ij"), axis=-1).reshape(-1, 3)


def probe_geometry(catalog: AssetCatalog, uid: str, probe: GeometryProbe) -> dict:
    asset = catalog.read(uid)
    mesh = instance_mesh(catalog, uid, probe.dimensions_m, require_volume=probe.capability != "graspable")
    measurements = {}
    accepted = False
    if probe.capability == "graspable":
        widths = mesh.extents[:2]
        accepted = bool(widths.min() + 2 * probe.clearance_m <= probe.maximum_gripper_opening_m)
        measurements = {"closing_widths_m": widths.tolist(),
                        "maximum_gripper_opening_m": probe.maximum_gripper_opening_m}
    else:
        points = _grid(probe.region_bounds_m, probe.samples_per_axis)
        if np.any(points < mesh.bounds[0] - probe.clearance_m) or np.any(points[:, :2] > mesh.bounds[1, :2]):
            raise ValueError("检查区域超出资产几何范围")
        occupied = mesh.contains(points)
        distances = trimesh.proximity.closest_point(mesh, points)[1]
        clear = ~occupied & (distances >= probe.clearance_m)
        origins = points
        directions = np.tile((0., 0., -1.), (len(points), 1))
        downward = mesh.ray.intersects_any(origins, directions)
        upward = mesh.ray.intersects_any(origins, -directions)
        measurements = {"sample_count": len(points), "occupied_samples": int(occupied.sum()),
                        "minimum_clearance_m": float(distances.min()),
                        "supported_samples": int(downward.sum()), "upward_blocked_samples": int(upward.sum())}
        accepted = bool(clear.all() and downward.all() and not upward.any())
        if probe.capability == "container":
            walls = []
            for direction in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
                walls.append(mesh.ray.intersects_any(origins, np.tile(direction, (len(points), 1))))
            measurements["side_enclosed_samples"] = [int(wall.sum()) for wall in walls]
            accepted = accepted and all(wall.all() for wall in walls)
            # 容器接触必须保留开口，单个 convex hull 会覆盖内部区域。
            accepted = accepted and probe.collision == "convexDecomposition"
    return {"schema_version": 1, "asset_uid": uid, "asset_sha256": asset.sha256,
            "probe": probe.model_dump(mode="json"), "probe_sha256": probe.digest(),
            "scope": "sampled_mesh_geometry", "geometry_accepted": bool(accepted),
            "measurements": measurements, "physical_task_verified": False,
            "source_sha256": file_digest(Path(__file__)), "trimesh_version": trimesh.__version__}


def verify_geometry_report(catalog: AssetCatalog, uid: str, report: dict, probe: GeometryProbe):
    if (report["asset_uid"] != uid or report["asset_sha256"] != catalog.read(uid).sha256
            or report["probe_sha256"] != probe.digest() or report["source_sha256"] != file_digest(Path(__file__))):
        raise ValueError("能力证据与资产、尺度、机器人或检查程序版本不一致")
    measured = probe_geometry(catalog, uid, probe)
    if measured != report:
        raise ValueError("能力证据与实际几何测量结果不一致")
    if not report["geometry_accepted"]:
        raise ValueError("资产几何未通过指定操作要求")
    return measured
