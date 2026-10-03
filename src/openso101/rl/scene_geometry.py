import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


def robot_collision_extras(stage, robot_path):
    body = stage.GetPrimAtPath(f"{robot_path}/gripper")
    group = stage.GetPrimAtPath(f"{robot_path}/gripper/collisions")
    if not body.IsValid() or not group.IsValid():
        raise ValueError("机器人缺少 gripper collision group")
    if not UsdPhysics.CollisionAPI(group).GetCollisionEnabledAttr().Get():
        raise ValueError("gripper collision group 必须启用")
    cache = UsdGeom.XformCache()
    body_inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)))
    records = []
    for prim in Usd.PrimRange(group, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or "/camera_mount/" not in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()
                or not np.isin(counts, [3, 4]).all() or counts.sum() != len(indices)
                or not len(indices) or indices.min() < 0 or indices.max() >= len(points)):
            raise ValueError("camera mount collision mesh 无效")
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)) @ body_inverse
        vertices = (np.column_stack((points, np.ones(len(points)))) @ transform)[:, :3]
        if not np.isfinite(vertices).all():
            raise ValueError("camera mount body transform 无效")
        polygons = np.split(indices, np.cumsum(counts)[:-1])
        records.append({"body": "gripper", "source_path": str(prim.GetPath()),
                        "vertices_body": vertices.tolist(), "polygons": [face.tolist() for face in polygons],
                        "native_approximation": UsdPhysics.MeshCollisionAPI(group).GetApproximationAttr().Get(),
                        "source": "native_USD_collision_mesh"})
    return records


def table_collision_geometry(stage, table_path, root_position, root_quaternion):
    root_position = np.asarray(root_position, dtype=float)
    root_quaternion = np.asarray(root_quaternion, dtype=float)
    if root_position.shape != (3,) or root_quaternion.shape != (4,):
        raise ValueError("机器人 root pose 形状不一致")
    if not np.isfinite(root_position).all() or not np.isfinite(root_quaternion).all():
        raise ValueError("机器人 root pose 需要有限值")
    rotation = Rotation.from_quat(root_quaternion, scalar_first=True)
    table = stage.GetPrimAtPath(table_path)
    if not table.IsValid():
        raise ValueError("原生场景缺少 Table prim")
    colliders = [prim for prim in Usd.PrimRange(table, Usd.TraverseInstanceProxies())
                 if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
    if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Mesh):
        raise ValueError("当前 portable 场景导出要求单个桌面 box collision mesh")
    collider = colliders[0]
    approximation = UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
    if approximation != "convexHull":
        raise ValueError("桌面 collision mesh 需要 convexHull")
    points = np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get(), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("桌面 collision mesh points 无效")
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(collider))
    if not np.isfinite(matrix).all():
        raise ValueError("桌面 box transform 需要有限值")
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    if (maximum <= minimum).any():
        raise ValueError("桌面 collision mesh 需要正数尺寸")
    if not np.all(np.minimum(np.abs(points-minimum), np.abs(points-maximum)) <= 1e-8):
        raise ValueError("桌面 collision mesh 需要 box 顶点")
    scales = np.linalg.norm(matrix[:3, :3], axis=1)
    if (scales <= 0).any():
        raise ValueError("桌面 box scale 必须为正数")
    axes = matrix[:3, :3] / scales[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-8, rtol=0) or np.linalg.det(axes) <= 0:
        raise ValueError("桌面 box transform 需要正数 scale 与正交坐标")
    local_axes = rotation.inv().apply(axes).T
    if np.max(np.abs(local_axes[2, :2])) > 1e-4 or local_axes[2, 2] <= 0:
        raise ValueError("当前任务要求水平桌面")
    world_center = (np.append((minimum+maximum)/2, 1) @ matrix)[:3]
    position = rotation.inv().apply(world_center-root_position)
    half_size = (maximum-minimum)/2 * scales
    top = position[2] + np.abs(local_axes[2]) @ half_size
    return {"type": "box", "position_root": position.tolist(),
            "quaternion_root": Rotation.from_matrix(local_axes).as_quat(scalar_first=True).tolist(),
            "half_size": half_size.tolist(), "top_height_root": float(top),
            "collider_paths": [str(collider.GetPath())], "approximation": approximation,
            "source": "native_USD_collision_mesh"}
