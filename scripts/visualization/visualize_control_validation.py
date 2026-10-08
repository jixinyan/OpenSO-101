import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FuncFormatter
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--initial-run", type=Path, required=True)
parser.add_argument("--fixed-run", type=Path, required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
paths = {"initial": args.initial_run / "demonstration_initialization.json",
         "old_control": args.initial_run / "final_control_audit.json",
         "fixed_control": args.fixed_run / "control_audit.json",
         "initial_evaluation": args.initial_run / "initial_policy_evaluation.json",
         "old_evaluation": args.initial_run / "evaluation_history.json",
         "fixed_evaluation": args.fixed_run / "evaluation_history.json"}
records = {name: json.loads(path.read_text()) for name, path in paths.items()}
if any(records[name]["dataset_sha256"] != records["initial"]["dataset_sha256"] for name in ("old_control", "fixed_control")):
    raise ValueError("动作误差图需要相同的实际来源样本")
labels = ("示范初始化", "PPO 配置 A", "PPO 配置 B")
title = "SO-101 Lift · 实际动作拟合与独立任务评估"
axis_titles = ("相同来源样本的动作误差", "各项条件满足的 episodes")
axis_labels = ("归一化动作 MSE", "episodes / 100")
legend = ("接近物体", "双侧接触", "物体抬升", "任务成功")
caption = "配置 A：动态统计、固定 learning rate 1e-4；配置 B：固定统计、adaptive KL、learning rate 1e-5。\n两项配置各完成八次 PPO 更新、49152 transitions。"
text = title + "".join(labels + axis_titles + axis_labels + legend) + caption
with TTFont(args.font) as font:
    available = font.getBestCmap()
    if any(ord(character) not in available for character in text if not character.isspace()):
        raise ValueError("图表字体需要包含全部使用的文字")
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
evaluations = (records["initial_evaluation"], records["old_evaluation"][-1], records["fixed_evaluation"][-1])
if any(len(record["episodes"]) != 100 for record in evaluations):
    raise ValueError("任务评估图需要每次包含 100 个实际 episodes")
mse = [records["initial"]["actor_mse"], records["old_control"]["model_mse"], records["fixed_control"]["model_mse"]]
if not np.isfinite(mse).all() or min(mse) <= 0:
    raise ValueError("图表需要有效的动作误差")
figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].bar(labels, mse, color=("#2563eb", "#64748b", "#059669"))
axes[0].set(yscale="log", title=axis_titles[0], ylabel=axis_labels[0])
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
for index, value in enumerate(mse):
    axes[0].text(index, value * 1.2, f"{value:.3e}", ha="center")
x = np.arange(3)
counts = {}
for index, (key, label) in enumerate(zip(("reached", "grasped", "lifted", "success"), legend, strict=True)):
    values = [sum(bool(episode[key]) for episode in record["episodes"]) for record in evaluations]
    counts[key] = values
    axes[1].bar(x + (index - 1.5) * .2, values, width=.2, label=label)
axes[1].set(xticks=x, xticklabels=labels, title=axis_titles[1], ylabel=axis_labels[1], ylim=(0, 100))
axes[1].legend(fontsize=9)
for axis in axes:
    axis.grid(axis="y", alpha=.2)
figure.suptitle(title)
figure.text(.5, .02, caption, ha="center", fontsize=9)
figure.tight_layout(rect=(0, .13, 1, 1))
args.output.mkdir(parents=True, exist_ok=False)
path = args.output / "control_validation.png"
figure.savefig(path, dpi=170)
plt.close(figure)
with Image.open(path) as image:
    image.verify()
report = {"status": "actual_control_validation_figure_verified", "action_mse": mse, "episode_counts": counts,
          "labels": labels, "figure_sha256": digest(path), "source_sha256": digest(Path(__file__)),
          "files": {name: digest(path) for name, path in paths.items()}, "task_success_verified": False}
with (args.output / "report.json").open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
