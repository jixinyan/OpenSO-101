import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from openso101.rl.config import digest

parser = argparse.ArgumentParser()
parser.add_argument("--reports", type=Path, required=True)
parser.add_argument("--scalars", type=Path, required=True)
parser.add_argument("--scene-trace", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
font = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
records = {}
for task in ("Lift", "PickPlace"):
    for condition in ("nominal", "randomized"):
        folder = args.reports / f"{task}_{condition}"
        record = json.loads((folder / "report.json").read_text())
        if digest(folder / "runtime.hdf5") != record["trace_sha256"]:
            raise ValueError("原生物理轨迹 SHA256 检查失败")
        records[f"{task}\n{condition}"] = record
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
labels = list(records)
axes[0].bar(labels, [item["maximum_physics_speed_rad_s"] for item in records.values()], color="#1b9e77")
axes[0].axhline(2, color="#c23b23", linestyle="--", label="检查标准 2 rad/s")
axes[0].set(ylabel="每个物理步骤测量的最高速度 / rad/s", ylim=(0, 2.3), title="实际关节速度")
axes[0].legend()
axes[1].bar(labels, [max(abs(value) for value in item["completed_episode_shaping_returns"])
                     for item in records.values()], color="#377eb8")
axes[1].set(ylabel="完整 episode 折扣 shaping 累计的绝对值", title="Reward 的完整 episode 检查")
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.suptitle("四个环境 × 500 个控制步骤 × 20 个物理步骤；源码 5b46a2f")
fig.tight_layout()
fig.savefig(args.output / "native_physics.png", dpi=160)
plt.close(fig)
scalars = json.loads(args.scalars.read_text())["scalars"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for axis, name, title in ((axes[0], "Loss/learning_rate", "训练记录中的 learning rate"),
                          (axes[1], "Loss/surrogate", "surrogate loss 的绝对值")):
    values = scalars[name]
    axis.plot([item["step"] for item in values], [abs(item["value"]) for item in values], "o-")
    axis.set(xlabel="PPO iteration", title=title, yscale="log")
    axis.grid(alpha=.2)
axes[0].axhline(1e-4, color="#1b9e77", linestyle="--", label="当前配置 0.0001")
axes[0].legend()
fig.suptitle("实际 TensorBoard 记录；源码 32b58c6")
fig.tight_layout()
fig.savefig(args.output / "training_records.png", dpi=160)
plt.close(fig)
with h5py.File(args.scene_trace) as trace:
    dt = float(trace.attrs["control_dt"])
    states = trace["states"][:]
    forces = trace["jaw_forces"][:]
    phases = trace["phase"][:]
    successes = trace["success"][:]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
time = np.arange(len(states)) * dt
for index in range(states.shape[1]):
    axes[0].plot(time, states[:, index, 0, 2], label=f"environment {index}")
    axes[1].plot(time, forces[:, index, 0].min(axis=-1))
    axes[2].plot(time, phases[:, index])
axes[0].set(title="实际 Apple 高度", ylabel="高度 / m", xlabel="时间 / s")
axes[0].legend()
axes[1].set(title="双侧接触力中的较小值", ylabel="接触力 / N", xlabel="时间 / s")
axes[1].axhline(.5, color="#c23b23", linestyle="--")
axes[2].set(title="TaskProgram 实际阶段", xlabel="时间 / s", ylabel="完成阶段数量", yticks=range(5))
fig.suptitle(f"四个环境的实际状态；任务成功次数：{int(successes.sum())}")
fig.tight_layout()
fig.savefig(args.output / "scene_program.png", dpi=160)
plt.close(fig)
report = {"source_sha256": digest(Path(__file__)), "scalar_sha256": digest(args.scalars),
          "scene_trace_sha256": digest(args.scene_trace),
          "runtime_reports": {name: item["trace_sha256"] for name, item in records.items()},
          "figures": {path.name: digest(path) for path in args.output.glob("*.png")}}
(args.output / "figures.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False))
