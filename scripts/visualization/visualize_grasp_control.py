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
parser.add_argument("--evaluation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
font = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
report = json.loads((args.task / "report.json").read_text())
geometry = json.loads((args.task / "geometry_contacts.json").read_text())
plan = json.loads((args.task / "plan.json").read_text())
if digest(args.task / "trajectory.hdf5") != report["trace_sha256"] or geometry["native_trace_sha256"] != report["trace_sha256"]:
    raise ValueError("可视化需要通过 SHA256 校验的实际轨迹")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
with h5py.File(args.task / "trajectory.hdf5", "r") as trace:
    times = np.arange(len(trace["phase"])) * report["control_dt"]
    for index in range(len(plan["environments"])):
        phases = trace["phase"][:, index]
        target_indices = np.where(phases == 2, 1, phases).clip(0, 2)
        targets = np.asarray([item["joint_position"] for item in plan["environments"][index]["targets"]])[target_indices]
        errors = np.linalg.norm(targets - trace["joint_position"][:, index, :5], axis=-1)
        axes[0, 0].plot(times, errors, label=f"环境 {index}")
        axes[0, 1].plot(times, trace["grasp_position_root"][:, index, 2] - trace["object_position_root"][:, index, 2],
                        label=f"环境 {index}")
axes[0, 0].axhline(.06, color="black", linestyle="--", label="阶段推进条件")
axes[0, 0].set(title="原生执行：关节目标误差", xlabel="控制时间（秒）", ylabel="五个关节误差的范数（rad）")
axes[0, 1].set(title="原生执行：抓取中心与物体的高度差", xlabel="控制时间（秒）", ylabel="高度差（米）")
required = [max(abs(point["holding_effort_nm"][1]) for point in item["planned_waypoints"]) for item in geometry["environments"]]
penetrations = [max((contact["penetration_m"] for point in item["planned_waypoints"]
                    for contact in point["contacts"] if "table" in contact["geometries"]), default=0.) * 1000
                for item in geometry["environments"]]
labels = [f"环境 {index}" for index in range(len(required))]
bars = axes[1, 0].bar(labels, required, color="#337ab7")
axes[1, 0].bar_label(bars, fmt="%.3f")
axes[1, 0].axhline(.712, color="#b22222", linestyle="--", label="静止控制范围 0.712 N·m")
axes[1, 0].set(title="MuJoCo 几何姿态：Pitch 保持力矩", ylabel="保持力矩（N·m）", ylim=(0, 1))
bars = axes[1, 1].bar(labels, penetrations, color="#d95f02")
axes[1, 1].bar_label(bars, fmt="%.2f")
axes[1, 1].set(title="MuJoCo 几何姿态：gripper 与桌面的交叉", ylabel="最大交叉深度（毫米）")
for axis in axes.flat:
    axis.grid(axis="y", alpha=.2)
for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
    axis.legend(fontsize=8)
fig.suptitle("实际 reset 的抓取检查：四个环境，任务成功 0/4\n原生动力学与 MuJoCo 几何检查分别保留验证范围", fontsize=13)
fig.tight_layout()
fig.savefig(args.output / "grasp_control.png", dpi=170)
plt.close(fig)
evaluation = json.loads(args.evaluation.read_text())[-1]
names = ("reached", "grasped", "held_above_table", "success")
counts = [sum(bool(item[name]) for item in evaluation["episodes"]) for name in names]
fig, axis = plt.subplots(figsize=(9, 4.5))
bars = axis.bar(["接近物体", "双侧接触", "持物抬升", "任务成功"], counts, color=["#337ab7", "#1b9e77", "#d95f02", "#b22222"])
axis.bar_label(bars, fmt="%.0f")
axis.set(title=f"基线训练 100 iterations 后的独立评估\n{len(evaluation['episodes'])} episodes，seed {evaluation['seed']}，{evaluation['completed_transitions']:,} transitions",
         ylabel="满足条件的 episode 数量", ylim=(0, 110))
axis.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(args.output / "baseline_progress.png", dpi=170)
plt.close(fig)
(args.output / "grasp_visualization_sources.json").write_text(json.dumps({
    "native_trace_sha256": report["trace_sha256"], "geometry_report_sha256": digest(args.task / "geometry_contacts.json"),
    "evaluation_sha256": digest(args.evaluation), "source_sha256": digest(Path(__file__)),
    "figures": {name: digest(args.output / name) for name in ("grasp_control.png", "baseline_progress.png")},
}, indent=2))
