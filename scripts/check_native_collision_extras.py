import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd

from openso101.rl.config import digest
from openso101.rl.scene_geometry import robot_collision_extras
from openso101.sim2sim.mujoco import build_model


parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--states", type=Path, required=True)
parser.add_argument("--robot-model", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
stage = Usd.Stage.Open(str(args.usd))
if stage is None:
    raise ValueError("需要有效 USD stage")
extras = robot_collision_extras(stage, str(stage.GetDefaultPrim().GetPath()))
if not extras:
    raise RuntimeError("原生资产缺少 camera mount")
states = json.loads(args.states.read_text())
metadata = states["planner_physics"]
original = build_model(args.robot_model, metadata, args.collision_bundle)
metadata["robot_collision_extras"] = extras
model = build_model(args.robot_model, metadata, args.collision_bundle)
for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
    if not np.allclose(getattr(model, name), getattr(original, name), atol=1e-12, rtol=0):
        raise RuntimeError("额外碰撞表示改变机器人质量、COM 或惯性")
added = [model.geom(index).name for index in range(model.ngeom)
         if model.geom(index).name.startswith("native_camera_mount_")]
if not added:
    raise RuntimeError("MuJoCo 缺少 camera mount convex parts")
report = {"status": "native_collision_extras_verified", "usd_sha256": digest(args.usd),
          "states_sha256": digest(args.states), "source_sha256": digest(Path(__file__)),
          "source_meshes": len(extras), "source_vertices": [len(item["vertices_body"]) for item in extras],
          "source_polygons": [len(item["polygons"]) for item in extras], "added_geometries": added,
          "inertial_properties_unchanged": True, "native_approximation": extras[0]["native_approximation"],
          "mujoco_approximation": "CoACD_convex_parts", "physics_equivalence_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
