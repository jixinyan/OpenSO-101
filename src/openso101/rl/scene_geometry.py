import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


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
    world = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]
    local = rotation.inv().apply(world - root_position)
    minimum, maximum = local.min(axis=0), local.max(axis=0)
    if (maximum <= minimum).any():
        raise ValueError("桌面 collision mesh 需要正数尺寸")
    if not np.all(np.minimum(np.abs(local-minimum), np.abs(local-maximum)) <= 1e-6):
        raise ValueError("桌面 collision mesh 需要与 robot root 坐标平行的 box")
    return {"type": "box", "position_root": ((minimum+maximum)/2).tolist(),
            "half_size": ((maximum-minimum)/2).tolist(), "top_height_root": float(maximum[2]),
            "collider_paths": [str(collider.GetPath())], "approximation": approximation,
            "source": "native_USD_collision_mesh"}
