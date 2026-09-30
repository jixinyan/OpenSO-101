import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
environment = dict(os.environ, OPENSO101_SKIP_ISAAC="1", PYTHONPATH=str(root / "src"),
                   TMPDIR=str(root / "outputs/tmp"))
for task in ("lift", "pick_place"):
    for mode in ("compare", "mujoco"):
        name = f"{task}_constrained_{args.run_id}_{mode}"
        command = [sys.executable, "-m", "openso101.cli.main", "sim2sim", mode,
                   "--policy", f"outputs/rl_progress/{task}_scene_geometry_oriented_verified",
                   "--robot-model", "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml",
                   "--episodes", "4", "--constrained-drive", "--output", f"outputs/rl_progress/{name}"]
        if mode == "compare":
            command += ["--steps", "500", "--recorded-pd", "--physics-components", "bodies", "gravity", "armature", "friction"]
        else:
            command += ["--recorded-physics"]
        with (root / f"outputs/rl_progress/{name}.log").open("x") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((root / f"outputs/rl_progress/{name}/report.json").read_text())
        if not report["actual_velocity_limits_verified"]:
            raise RuntimeError("实际关节速度约束检查未通过")
        print(f"{name}: actual_velocity_limits_verified", flush=True)
