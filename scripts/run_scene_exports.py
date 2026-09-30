import argparse
import json
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for task, task_id in (("lift", "OpenSO101-Lift-v0"), ("pick_place", "OpenSO101-PickPlace-v0")):
    output = root / f"outputs/rl_progress/{task}_scene_geometry_{args.run_id}"
    command = [sys.executable, "-m", "openso101.cli.main", "rl", "export", "--task", task_id,
               "--checkpoint", f"outputs/rl_progress/{task}_grasp_v2_snapshot_50", "--output", str(output),
               "--num-envs", "4", "--validation-steps", "500", "--headless"]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    validation = json.loads((output / "validation.json").read_text())
    metadata = json.loads((output / "policy.json").read_text())
    if (validation["status"] != "portable_policy_numerically_verified_in_isaac"
            or "table_geometry" not in metadata or validation["validation_steps"] != 500 or validation["num_envs"] != 4):
        raise RuntimeError("原生推理与场景导出结果缺少验证")
    print(f"{task}: 保存模型推理与场景导出检查完成", flush=True)
