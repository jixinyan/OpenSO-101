import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from .snapshot import snapshot


def evaluate_snapshot(run, checkpoint, iteration, config):
    destination = run / "evaluations" / f"iteration_{iteration:06d}"
    destination.parent.mkdir(exist_ok=True)
    snapshot(Namespace(run=str(run), checkpoint=checkpoint.name, output=str(destination)))
    from .config import CheckpointMeta

    meta = CheckpointMeta.read(destination)
    command = [sys.executable, "-m", "openso101.cli.main", "rl", "eval", "--task", meta.task_id,
               "--checkpoint", str(destination), "--n-episodes", str(config.evaluation_episodes),
               "--num-envs", "64", "--seed", str(config.seed + 10000), "--headless"]
    with (destination / "evaluation.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    reports = list(destination.glob("evaluation-*.json"))
    if len(reports) != 1:
        raise RuntimeError("独立评估需要一份完整的报告")
    result = json.loads(reports[0].read_text())
    if len(result["episodes"]) != config.evaluation_episodes:
        raise RuntimeError("独立评估的 episode 数量不匹配")
    result["iteration"] = iteration
    result["source_report"] = str(reports[0].relative_to(run))
    history_path = run / "evaluation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(result)
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    best_path = run / "best_evaluation.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or result["success_rate"] > best["success_rate"]:
        best_path.write_text(json.dumps(result, indent=2) + "\n")
        shutil.copy2(checkpoint, run / "model_best.pt")
    converged = len(history) >= 3 and all(item["success_rate"] >= .9 for item in history[-3:])
    (run / "convergence.json").write_text(json.dumps({
        "seed": config.seed, "task_id": meta.task_id, "episodes_per_evaluation": config.evaluation_episodes,
        "required_consecutive_evaluations": 3, "threshold": .9,
        "seed_converged": converged, "last_iteration": iteration,
        "multi_seed_acceptance_verified": False,
    }, indent=2) + "\n")
    return converged
