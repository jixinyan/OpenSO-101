import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OpenSO101-Lift-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
if args.num_envs <= 0:
    raise ValueError("num-envs 必须为正数")
args.task_profile = "grasp_v2"
args.seed = 42
args.with_cameras = False

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    import torch

    from openso101.rl.execution import build_environment
    from openso101.rl.config import digest
    from openso101.rl.scene_geometry import table_collision_geometry

    env = build_environment(args, training=False)
    env.reset()
    runtime = env.unwrapped
    robot = runtime.scene["robot"]
    obj = runtime.scene["object"]
    stage = runtime.sim.stage
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    colliders = []
    materials = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/env_0/" not in path:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            colliders.append({"path": path, "type": prim.GetTypeName(), "schemas": list(prim.GetAppliedSchemas()),
                              "world_min_m": list(box.GetMin()), "world_max_m": list(box.GetMax()),
                              "attributes": {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                                             if attr.GetName().startswith(("physics:", "physx")) and "buffer" not in attr.GetName()}})
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            materials.append({"path": path, "attributes": {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                                                           if attr.GetName().startswith(("physics:", "physx"))}})
    print(json.dumps({"robot_root_position": robot.data.root_pos_w[0].tolist(), "robot_root_quaternion": robot.data.root_quat_w[0].tolist(),
                      "table_colliders": [item for item in colliders if "/Table/" in item["path"]]}), flush=True)
    table_geometry = table_collision_geometry(stage, "/World/envs/env_0/Table",
                                              robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
    physics = {}
    for name, asset in (("robot", robot), ("object", obj)):
        view = asset.root_physx_view
        physics[name] = {field: getattr(view, f"get_{field}")().cpu().numpy().tolist()
                         for field in ("masses", "inertias", "coms", "material_properties")}
    fixed_positions = robot.data.joint_pos.clone()
    initial_roots = robot.data.root_pos_w.clone()
    initial_quaternions = robot.data.root_quat_w.clone()
    root_offsets = initial_roots - runtime.scene.env_origins
    cloning_error = float(torch.max(torch.abs(root_offsets-root_offsets[0])))
    if cloning_error > 1e-5:
        raise RuntimeError("克隆环境的机器人 root 位置不一致")
    robot.set_joint_position_target(fixed_positions)
    positions = []
    root_drift = 0.
    root_rotation_drift = 0.
    for step in range(200):
        runtime.scene.write_data_to_sim()
        runtime.sim.step(render=False)
        runtime.scene.update(runtime.physics_dt)
        if not torch.isfinite(obj.data.root_state_w).all():
            raise RuntimeError("原生桌面实验产生无效状态")
        positions.append(obj.data.root_pos_w.cpu().numpy().tolist())
        root_drift = max(root_drift, float(torch.max(torch.abs(robot.data.root_pos_w-initial_roots))))
        root_rotation_drift = max(root_rotation_drift, float(torch.max(torch.abs(robot.data.root_quat_w-initial_quaternions))))
    if max(root_drift, root_rotation_drift) > 1e-6:
        raise RuntimeError("固定机器人 root 在运行中发生移动")
    report = {"status": "native_scene_geometry_and_settling_checked", "task": args.task,
              "physics_dt": runtime.physics_dt, "control_dt": runtime.step_dt,
              "robot_root_world": robot.data.root_pos_w.cpu().numpy().tolist(),
              "table_geometry": table_geometry,
              "num_envs": args.num_envs, "robot_root_quaternion_world": initial_quaternions.cpu().numpy().tolist(),
              "robot_root_env_origin_offsets": root_offsets.cpu().numpy().tolist(),
              "root_cloning_position_error_m": cloning_error,
              "fixed_root_drift_m": root_drift, "fixed_root_quaternion_drift": root_rotation_drift,
              "colliders": colliders, "materials": materials, "physics": physics,
              "object_positions_world": positions, "settled_cube_center_world_z_m": float(np.asarray(positions)[-20:, :, 2].mean()),
              "source_code_sha256": digest(Path(__file__)), "task_success_verified": False}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "robot_root_world", "settled_cube_center_world_z_m")}))
finally:
    if env is not None:
        env.close()
app.close()
