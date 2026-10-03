import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=Path, required=True)
parser.add_argument("--evaluation", type=Path, action="append", required=True)
parser.add_argument("--label", action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if len(args.evaluation) != len(args.label):
    raise ValueError("每份评估需要对应名称")
args.output.mkdir(parents=True, exist_ok=True)
font = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
report = json.loads((args.task / "report.json").read_text())
if digest(args.task / "trajectory.hdf5") != report["trace_sha256"]:
    raise ValueError("原生轨迹的 SHA256 不一致")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
with h5py.File(args.task / "trajectory.hdf5", "r") as trace:
    times = (np.arange(len(trace["phase"])) + 1) * report["control_dt"]
    for index, environment in enumerate(report["environments"]):
        selected = trace["active"][:, index].astype(bool)
        label = f"环境 {index} · {'成功' if environment['success'] else '未完成'}"
        axes[0, 0].plot(times[selected], trace["object_position_root"][selected, index, 2], label=label)
        axes[0, 1].plot(times[selected], np.minimum(*trace["jaw_forces"][selected, index].T), label=label)
        axes[1, 0].plot(times[selected], trace["phase"][selected, index], label=label)
        velocities = trace["physics_steps/joint_velocity"][:, index]
        physics_times = (np.arange(len(velocities)) + 1) * report["physics_dt"]
        axes[1, 1].plot(physics_times[physics_times <= times[selected][-1]],
                        np.abs(velocities).max(axis=-1)[physics_times <= times[selected][-1]], label=label)
axes[0, 0].set(title="物体的实际高度", ylabel="robot root frame 高度（米）")
axes[0, 1].axhline(.5, color="black", linestyle="--", label="双侧接触阈值")
axes[0, 1].set(title="两个夹爪接触力的较小值", ylabel="接触力（N）")
axes[1, 0].set(title="脚本控制的实际阶段", yticks=[-1, 0, 1, 2, 3],
                yticklabels=["准备", "接近", "抓取", "闭合", "提升"])
axes[1, 1].set(title="逐物理步骤的最高关节速度", ylabel="速度（rad/s）")
for axis in axes.flat:
    axis.set_xlabel("执行时间（秒）")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
fig.suptitle(f"{report['task_profile']} 原生脚本控制 · 成功 {report['successes']}/{report['episodes']}\n"
             "物体位置、接触力和关节速度均来自实际仿真记录", fontsize=13)
fig.tight_layout()
fig.savefig(args.output / "native_position_grasp.png", dpi=170)
plt.close(fig)

fig, axis = plt.subplots(figsize=(11, 5))
names = ("reached", "grasped", "held_above_table", "success")
width = .8 / len(args.evaluation)
sources = []
for index, (path, label) in enumerate(zip(args.evaluation, args.label, strict=True)):
    evaluation = json.loads(path.read_text())[-1]
    episodes = evaluation["episodes"]
    if len(episodes) != 100:
        raise ValueError("比较图需要完整 100 episodes 独立评估")
    counts = [sum(bool(item[name]) for item in episodes) for name in names]
    bars = axis.bar(np.arange(len(names)) + index * width, counts, width,
                    label=f"{label} · {evaluation['completed_transitions']:,} transitions")
    axis.bar_label(bars, fontsize=8)
    sources.append({"label": label, "report_sha256": digest(path),
                    "checkpoint_sha256": evaluation["checkpoint_sha256"], "counts": dict(zip(names, counts, strict=True))})
axis.set(title="独立任务评估 · 每个模型 100 episodes", ylabel="满足条件的 episode 数量", ylim=(0, 112),
         xticks=np.arange(len(names)) + width * (len(args.evaluation) - 1) / 2,
         xticklabels=["接近物体", "双侧接触", "持物抬升", "任务成功"])
axis.legend(fontsize=9)
axis.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(args.output / "independent_evaluations.png", dpi=170)
plt.close(fig)
(args.output / "position_visualization_sources.json").write_text(json.dumps({
    "native_report_sha256": digest(args.task / "report.json"), "native_trace_sha256": report["trace_sha256"],
    "evaluations": sources, "source_sha256": digest(Path(__file__)),
    "figures": {name: digest(args.output / name) for name in ("native_position_grasp.png", "independent_evaluations.png")},
}, indent=2))
