import argparse
import json
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("paths", type=Path, nargs="+")
args = parser.parse_args()
for path in args.paths:
    if path.name == "evaluation_history.json":
        for record in json.loads(path.read_text()):
            print(json.dumps({key: record[key] for key in
                              ("task", "seed", "iteration", "success_rate", "progress_rates",
                               "checkpoint_sha256", "completed_transitions")}))
    else:
        report = json.loads((path / "report.json").read_text())
        with h5py.File(path / "runtime.hdf5") as trace:
            velocity = trace["physics_steps/joint_velocity"][:]
            physics_step, environment, joint = np.unravel_index(np.abs(velocity).argmax(), velocity.shape)
            control_step = physics_step // round(report["control_dt"] / report["physics_dt"])
            report["maximum_speed_context"] = {
                "physics_step": int(physics_step), "environment": int(environment), "joint_index": int(joint),
                "joint_position_before": trace["joint_position_before"][control_step, environment].tolist(),
                "raw_action": trace["raw_action"][control_step, environment].tolist(),
                "velocity": velocity[physics_step, environment].tolist(),
            }
        print(json.dumps(report))
