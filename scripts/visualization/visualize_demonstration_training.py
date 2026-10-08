import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
from PIL import Image

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
initialization = json.loads((args.run / "demonstration_initialization.json").read_text())
pretrain = [json.loads(line) for line in (args.run / "demonstration_pretrain.jsonl").read_text().splitlines()]
first = json.loads((args.run / "initial_policy_evaluation.json").read_text())
history = json.loads((args.run / "evaluation_history.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].semilogy([item["epoch"] for item in pretrain], [item["actor_mse"] for item in pretrain], color="#2563eb")
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.0e}"))
axes[0].yaxis.set_minor_formatter(NullFormatter())
axes[0].set(title="实际成功轨迹的动作拟合", xlabel="监督训练 epochs", ylabel="bounded action MSE")
names = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
axes[1].bar(names, initialization["rmse_per_action"], color="#059669")
axes[1].tick_params(axis="x", rotation=25)
axes[1].set(title="初始化模型在来源轨迹上的误差", ylabel="归一化动作 RMSE")
for axis in axes:
    axis.grid(axis="y", alpha=.2)
fig.suptitle(f"SO-101 Lift：{initialization['frames']} 个真实成功轨迹样本", fontsize=15)
fig.tight_layout()
fig.savefig(args.output / "demonstration_fit.png", dpi=160)
plt.close(fig)

evaluations = [first, *history]
labels = ["示范初始化", *[f"PPO iteration {item['iteration']}" for item in history]]
fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
for axis, key, title in zip(axes, ("reached", "grasped", "lifted", "success"),
                            ("接近物体", "双侧接触", "物体抬升", "任务成功"), strict=True):
    values = [sum(bool(episode[key]) for episode in report["episodes"]) for report in evaluations]
    if any(len(report["episodes"]) != 100 for report in evaluations):
        raise ValueError("图表要求每次独立评估包含 100 个真实 episode")
    axis.bar(labels, values, color=["#2563eb", *["#059669"] * len(history)])
    for index, value in enumerate(values):
        axis.text(index, value + 2, f"{value}/100", ha="center")
    axis.set(title=title, ylabel="episodes", ylim=(0, 105))
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=.2)
    if key == "success":
        axis.axhline(90, linestyle="--", color="#64748b", label="验收要求")
        axis.legend()
fig.suptitle("保存模型的实际独立评估", fontsize=15)
fig.tight_layout()
fig.savefig(args.output / "independent_evaluation.png", dpi=160)
plt.close(fig)
figures = {}
for path in args.output.glob("*.png"):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        figures[path.name] = {"sha256": digest(path), "width": image.width, "height": image.height}
result = {"status": "actual_demonstration_and_evaluation_figures_verified", "figures": figures,
          "source_sha256": digest(Path(__file__)),
          "files": {name: digest(args.run / name) for name in ("demonstration_initialization.json",
            "demonstration_pretrain.jsonl", "initial_policy_evaluation.json", "evaluation_history.json")},
          "task_success_threshold_verified": all(item["success_rate"] >= .9 for item in evaluations)}
(args.output / "figures.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
