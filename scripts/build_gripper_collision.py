import argparse
import importlib.metadata
import json
from pathlib import Path

import coacd
import numpy as np
import trimesh

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
assets = root / "outputs/so-arm100/Simulation/SO101/assets"
parameters = {"threshold": .01, "preprocess_mode": "auto", "preprocess_resolution": 100,
              "resolution": 10000, "mcts_iterations": 200, "mcts_max_depth": 4, "seed": 42}
coacd.set_log_level("warn")
meshes = {}
for name in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
    source = assets / f"{name}.stl"
    mesh = trimesh.load_mesh(source)
    if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
        raise ValueError("夹爪 STL 需要有效的 triangle mesh")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **parameters)
    if len(parts) < 2:
        raise RuntimeError("夹爪碰撞生成需要多个 convex parts")
    records = []
    for index, (vertices, faces) in enumerate(parts):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if not part.is_convex or not part.is_watertight or not np.isfinite(vertices).all():
            raise RuntimeError("夹爪 convex part 无效")
        path = args.output / f"{name}_{index:03d}.obj"
        part.export(path)
        records.append({"file": path.name, "sha256": digest(path), "vertices": len(vertices), "faces": len(faces)})
    meshes[name] = {"source_sha256": digest(source), "source_vertices": len(mesh.vertices), "parts": records}
    print(f"{name}: {len(parts)} convex parts", flush=True)
manifest = {"schema_version": 1, "generator": "CoACD", "version": importlib.metadata.version("coacd"),
            "parameters": parameters, "threshold_unit": "normalized_mesh_extent", "meshes": meshes,
            "source_code_sha256": digest(Path(__file__))}
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
