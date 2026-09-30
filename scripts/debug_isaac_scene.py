import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="OpenSO101-Lift-v0")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.num_envs = 1
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
                                             if attr.GetName().startswith(("physics:", "physx"))}})
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            materials.append({"path": path, "attributes": {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                                                           if attr.GetName().startswith(("physics:", "physx"))}})
    physics = {}
    for name, asset in (("robot", robot), ("object", obj)):
        view = asset.root_physx_view
        physics[name] = {field: getattr(view, f"get_{field}")().cpu().numpy().tolist()
                         for field in ("masses", "inertias", "coms", "material_properties")}
    fixed_positions = robot.data.joint_pos.clone()
    robot.set_joint_position_target(fixed_positions)
    positions = []
    for step in range(200):
        runtime.scene.write_data_to_sim()
        runtime.sim.step(render=False)
        runtime.scene.update(runtime.physics_dt)
        if not torch.isfinite(obj.data.root_state_w).all():
            raise RuntimeError("原生桌面实验产生无效状态")
        positions.append(obj.data.root_pos_w.cpu().numpy().tolist())
    report = {"status": "native_scene_geometry_and_settling_checked", "task": args.task,
              "physics_dt": runtime.physics_dt, "control_dt": runtime.step_dt,
              "robot_root_world": robot.data.root_pos_w.cpu().numpy().tolist(),
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
