import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if len(args.folder) != 4:
    raise ValueError("比较需要四个实际 PPO backend")
args.output.mkdir(parents=True, exist_ok=False)
tags = {"rsl_rl": ("Train/mean_reward", "Logged PPO iteration"),
        "sb3": ("rollout/ep_rew_mean", "Logged transitions"),
        "skrl": ("Reward / Total reward (mean)", "Logged vector step"),
        "rl_games": ("rewards/step", "Logged transitions")}
records = []
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for folder, axis in zip(args.folder, axes.flat, strict=True):
    report = json.loads((folder / "report.json").read_text())
    if (report["algo"] != "ppo" or report["task_profile"] != "grasp_v4" or report["iterations"] != 50
            or report["training_transitions"] != 51200 or report["independent_evaluation_episodes"] != 100):
        raise ValueError("学习曲线比较需要实际完成的指定训练与独立评估")
    evaluations = list((folder / "run").glob("evaluation-*.json"))
    events = list((folder / "run").rglob("events.out.tfevents.*"))
    if len(evaluations) != 1 or len(events) != 1:
        raise ValueError("每个 backend 需要一份最终独立评估与一份实际训练事件文件")
    if digest(evaluations[0]) != report["evaluation_sha256"]:
        raise ValueError("独立评估的 SHA256 不一致")
    evaluation = json.loads(evaluations[0].read_text())
    if evaluation["checkpoint_sha256"] != report["checkpoint_sha256"] or len(evaluation["episodes"]) != 100:
        raise ValueError("评估模型或 episode 数量与实际报告不一致")
    tag, unit = tags[report["backend"]]
    events_data = EventAccumulator(str(events[0]), size_guidance={"scalars": 0}).Reload().Scalars(tag)
    steps = np.asarray([entry.step for entry in events_data])
    values = np.asarray([entry.value for entry in events_data])
    if not len(values) or not np.isfinite(values).all() or (np.diff(steps) <= 0).any():
        raise ValueError("实际训练曲线需要有效数值和连续递增的日志步骤")
    axis.plot(steps, values)
    axis.set(title=f"{report['backend']} · {report['action_distribution']}", xlabel=unit, ylabel="Logged mean episode return")
    axis.grid(alpha=.2)
    counts = {name: sum(bool(episode[name]) for episode in evaluation["episodes"])
              for name in ("reached", "grasped", "held_above_table", "success")}
    records.append({"backend": report["backend"], "report_sha256": digest(folder / "report.json"),
                    "checkpoint_sha256": report["checkpoint_sha256"], "evaluation_sha256": report["evaluation_sha256"],
                    "events_sha256": digest(events[0]), "counts": counts,
                    "training_transitions": report["training_transitions"], "tag": tag, "progress_unit": unit,
                    "curve": [{"step": int(step), "value": float(value)} for step, value in zip(steps, values, strict=True)]})
if {record["backend"] for record in records} != set(tags):
    raise ValueError("需要四个不同的 PPO backend")
fig.suptitle("Actual grasp_v4 PPO runs · 51,200 transitions each\nEach backend uses its own episode averaging window and log units")
fig.tight_layout()
fig.savefig(args.output / "backend_learning_curves.png", dpi=170)
plt.close(fig)
fig, axis = plt.subplots(figsize=(12, 5))
names = ("reached", "grasped", "held_above_table", "success")
for index, record in enumerate(records):
    bars = axis.bar(np.arange(4) + index * .2, [record["counts"][name] for name in names], .2, label=record["backend"])
    axis.bar_label(bars, fontsize=8)
axis.set(title="Final independent evaluation · 100 episodes per model", ylabel="Episodes meeting the condition",
         ylim=(0, 110), xticks=np.arange(4) + .3, xticklabels=["Reached object", "Bilateral contact", "Held above table", "Task success"])
axis.grid(axis="y", alpha=.2)
axis.legend()
fig.tight_layout()
fig.savefig(args.output / "backend_independent_evaluation.png", dpi=170)
plt.close(fig)
result = {"status": "actual_backend_curves_recorded", "backends": records, "source_sha256": digest(Path(__file__)),
          "figures": {name: digest(args.output / name) for name in (
              "backend_learning_curves.png", "backend_independent_evaluation.png")}, "task_success_threshold_verified": False}
(args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "counts": {record["backend"]: record["counts"] for record in records}}, indent=2), flush=True)
