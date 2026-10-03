import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=Path, action="append", required=True)
parser.add_argument("--baseline-directory", type=Path, required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--font", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
font_manager.findfont(args.font, fallback_to_default=False)
if args.num_envs <= 0 or len(args.folder) != 3:
    raise ValueError("曲线需要三个实际训练目录与有效环境数量")
args.output.mkdir(parents=True, exist_ok=False)
tags = {
    "Train/mean_reward": ("平均 episode return", "日志记录的 return"),
    "Metrics/object_pose/position_error": ("物体到任务目标的距离", "米"),
    "Policy/mean_noise_std": ("动作探索强度", "Gaussian 标准差"),
    "Loss/value_function": ("Value function loss", "日志记录的 loss"),
    "Perf/total_fps": ("训练速度", "每秒 transitions"),
    "Episode_Termination/success": ("训练中的任务成功比例", "日志记录的成功比例"),
}
colors = {42: "#2563eb", 43: "#e87919", 44: "#059669"}
records = []
for folder in args.folder:
    config = json.loads((folder / "train.json").read_text())
    seed = config["seed"]
    if seed not in colors or config["backend"] != "rsl_rl" or config["algo"] != "ppo":
        raise ValueError("输入需要 seeds 42、43、44 的实际 RSL PPO 记录")
    event_files = list(folder.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError("每个训练目录需要一份实际 TensorBoard 事件文件")
    events = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0}).Reload()
    resume = json.loads((folder / "resume.json").read_text())
    curves = {}
    samples_per_update = args.num_envs * config["rollout_steps"]
    for tag in tags:
        entries = events.Scalars(tag)
        steps = np.asarray([entry.step for entry in entries], dtype=np.int64)
        values = np.asarray([entry.value for entry in entries])
        if not len(entries) or not np.isfinite(values).all() or (np.diff(steps) <= 0).any():
            raise ValueError("实际曲线需要有效数值与递增的 iteration")
        if resume["prior_transitions"] != int(steps[0]) * samples_per_update:
            raise ValueError("事件起始 iteration 与父模型累计 transitions 不一致")
        curves[tag] = [{"iteration": int(entry.step),
                        "completed_transitions": (int(entry.step) + 1) * samples_per_update,
                        "value": float(entry.value), "wall_time": entry.wall_time}
                       for entry in entries]
    baseline = args.baseline_directory / f"OpenSO101-Lift-v0_seed_{seed}" / "evaluation_history.json"
    current = folder / "evaluation_history.json"
    evaluations = []
    for source in (baseline, current):
        for evaluation in json.loads(source.read_text()):
            episodes = evaluation["episodes"]
            if (len(episodes) != 100 or evaluation["task_profile"] != "grasp_v4"
                    or evaluation["seed"] != seed + 10000):
                raise ValueError("独立评估需要指定任务、seed 和完整的 100 episodes")
            counts = {name: sum(bool(episode[name]) for episode in episodes)
                      for name in ("reached", "grasped", "lifted", "held_above_table", "success")}
            if counts["success"] / 100 != evaluation["success_rate"]:
                raise ValueError("评估成功率与实际逐 episode 记录不一致")
            evaluations.append({"iteration": evaluation["iteration"],
                                "completed_transitions": evaluation["completed_transitions"],
                                "counts": counts, "checkpoint_sha256": evaluation["checkpoint_sha256"],
                                "source_sha256": digest(source)})
    evaluations.sort(key=lambda entry: entry["completed_transitions"])
    if any(right["completed_transitions"] <= left["completed_transitions"]
           for left, right in zip(evaluations, evaluations[1:])):
        raise ValueError("独立评估需要不同模型与递增的实际 transitions")
    records.append({"seed": seed, "run": folder.name, "num_envs": args.num_envs,
                    "prior_transitions": resume["prior_transitions"],
                    "events_sha256": digest(event_files[0]), "train_sha256": digest(folder / "train.json"),
                    "resume_sha256": digest(folder / "resume.json"),
                    "curves": curves, "evaluations": evaluations})
if {record["seed"] for record in records} != set(colors):
    raise ValueError("需要三个不同的训练 seed")

plt.rcParams.update({"font.size": 10, "font.family": args.font, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for axis, (tag, (title, ylabel)) in zip(axes.flat, tags.items(), strict=True):
    for record in records:
        curve = record["curves"][tag]
        x = np.asarray([entry["completed_transitions"] / 1e6 for entry in curve])
        y = np.asarray([entry["value"] for entry in curve])
        color = colors[record["seed"]]
        axis.plot(x, y, color=color, alpha=.22, linewidth=.8)
        # 保留全部原始点，移动平均只包含当前点与前面的实际记录。
        smooth = np.asarray([y[max(0, index - 9):index + 1].mean() for index in range(len(y))])
        axis.plot(x, smooth, color=color, linewidth=1.8, label=f"Seed {record['seed']}")
    axis.set(title=title, xlabel="累计训练 transitions（百万）", ylabel=ylabel)
    axis.grid(alpha=.18)
    if tag == "Episode_Termination/success":
        axis.set_ylim(-.005, max(.02, axis.get_ylim()[1]))
axes[0, 0].legend()
fig.suptitle("SO-101 Lift / grasp_v4 / RSL PPO\n当前继续训练的记录：原始数值与最近 10 次更新的移动平均", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(args.output / "training_curves.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
for axis, name, title in zip(axes, ("reached", "grasped", "lifted", "success"),
                             ("接近物体", "双侧接触", "物体抬升", "任务成功"), strict=True):
    for record in records:
        x = [entry["completed_transitions"] / 1e6 for entry in record["evaluations"]]
        y = [entry["counts"][name] for entry in record["evaluations"]]
        axis.plot(x, y, "o-", color=colors[record["seed"]], label=f"Seed {record['seed']}：最新 {y[-1]}%")
    axis.set(title=title, xlabel="累计训练 transitions（百万）", ylabel="Episodes（%）", ylim=(-5, 105))
    axis.grid(alpha=.18)
    if name == "success":
        axis.axhline(90, color="#64748b", linestyle="--", linewidth=1)
        axis.text(.04, .86, "验收要求：90%", transform=axis.transAxes, color="#64748b", fontsize=9)
    axis.legend(loc="center right", fontsize=8)
fig.suptitle("独立确定性评估：每个 checkpoint 运行 100 episodes\n圆点表示实际评估，连线连接已记录的 checkpoint", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .85))
fig.savefig(args.output / "independent_evaluation.png", dpi=160)
plt.close(fig)
figures = {}
for name in ("training_curves.png", "independent_evaluation.png"):
    with Image.open(args.output / name) as image:
        image.verify()
    with Image.open(args.output / name) as image:
        figures[name] = {"sha256": digest(args.output / name), "width": image.width, "height": image.height}
report = {"status": "actual_scoped_training_curves_recorded", "created_at": datetime.now(UTC).isoformat(),
          "source_sha256": digest(Path(__file__)), "smoothing": "trailing_10_logged_updates",
          "scope": "current_resumed_training_and_preserved_independent_evaluations",
          "runs": records, "figures": figures,
          "task_success_threshold_verified": all(record["evaluations"][-1]["counts"]["success"] >= 90
                                                 for record in records)}
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "latest": [
    {"seed": record["seed"], "iteration": record["curves"]["Train/mean_reward"][-1]["iteration"],
     "evaluation": record["evaluations"][-1]} for record in records]}, indent=2))
