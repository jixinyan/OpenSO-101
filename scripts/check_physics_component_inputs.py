import json
import os
from pathlib import Path
import subprocess
import sys

from openso101.rl.config import digest


root = Path(__file__).resolve().parents[1]
output = root / "outputs/rl_progress/physics_component_inputs_report.json"
if output.exists():
    raise FileExistsError(output)
environment = dict(os.environ, OPENSO101_SKIP_ISAAC="1", PYTHONPATH=str(root / "src"),
                   TMPDIR=str(root / "outputs/tmp"))
command = [sys.executable, "-m", "openso101.cli.main", "sim2sim", "compare",
           "--robot-model", "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml",
           "--episodes", "4", "--steps", "500", "--recorded-pd", "--velocity-servo"]
cases = (
    ("existing_output", "lift_body_physics_verified_50", "lift_physics_components_paired_baseline", [], "FileExistsError"),
    ("duplicate_components", "lift_body_physics_verified_50", "physics_components_duplicate_rejected",
     ["--physics-components", "bodies", "bodies"], "physics-components 不允许重复"),
    ("missing_recorded_bodies", "lift_physics_portable_50", "physics_components_old_source_rejected",
     ["--physics-components", "bodies"], "所选实际物理参数缺少源字段"),
)
records = []
for name, policy, destination, arguments, expected in cases:
    target = root / "outputs/rl_progress" / destination
    existed = target.exists()
    result = subprocess.run([*command, "--policy", f"outputs/rl_progress/{policy}",
                             "--output", str(target), *arguments], cwd=root, env=environment,
                            capture_output=True, text=True)
    assert result.returncode != 0 and expected in result.stderr
    assert target.exists() == existed
    records.append({"name": name, "rejected_verified": True, "returncode": result.returncode,
                    "output_creation_unchanged_verified": True, "expected_error": expected})
output.write_text(json.dumps({"checks": records, "script_sha256": digest(Path(__file__))},
                             ensure_ascii=False, indent=2))
print(output)
