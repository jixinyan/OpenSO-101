import os
import argparse
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
conditions = {"baseline": [], "bodies": ["bodies"], "gravity": ["gravity"],
              "armature": ["armature"], "friction": ["friction"],
              "combined": ["bodies", "gravity", "armature", "friction"]}
environment = dict(os.environ, OPENSO101_SKIP_ISAAC="1", PYTHONPATH=str(root / "src"),
                   TMPDIR=str(root / "outputs/tmp"))
for task in ("lift", "pick_place"):
    for condition, components in conditions.items():
        name = f"{task}_physics_components_{args.run_id}_{condition}"
        command = [sys.executable, "-m", "openso101.cli.main", "sim2sim", "compare",
                   "--policy", f"outputs/rl_progress/{task}_body_physics_verified_50",
                   "--robot-model", "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml",
                   "--episodes", "4", "--steps", "500", "--recorded-pd", "--velocity-servo",
                   "--output", f"outputs/rl_progress/{name}"]
        if components:
            command += ["--physics-components", *components]
        with (root / f"outputs/rl_progress/{name}.log").open("w") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{name}: completed", flush=True)
