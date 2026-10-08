import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
from PIL import Image

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--paired", type=Path, required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = json.loads((args.native / "report.json").read_text())
paired = json.loads(args.paired.read_text())
if (digest(args.native / "trajectory.hdf5") != source["trace_sha256"]
        or paired["source_trace_sha256"] != source["trace_sha256"]
        or paired["initial_observation_error"] != 0.
        or paired["task"] != source["task"] or paired["task_profile"] != source["task_profile"]):
    raise ValueError("图表需要相同初始状态与已校验的实际控制轨迹")
args.output.mkdir(parents=True, exist_ok=False)
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(2, 2, figsize=(12, 7))
curves = {}
with h5py.File(args.native / "trajectory.hdf5") as stream:
    selected = np.flatnonzero(stream["active"][:, 0])
    if len(selected) != source["environments"][0]["control_steps"] or not source["environments"][0]["success"]:
        raise ValueError("来源需要完整的实际成功 episode")
    curves["scripted"] = {"jaw_target": stream["joint_targets"][selected, 0, -1],
                          "jaw_actual": stream["joint_position"][selected, 0, -1],
                          "bilateral_force": stream["jaw_forces"][selected, 0].min(-1),
                          "object_height": stream["object_position_root"][selected, 0, 2],
                          "return": stream["weighted_reward"][selected, 0].sum(-1).cumsum()}
steps = paired["trajectory"]
if len(steps) != paired["steps"]:
    raise ValueError("模型轨迹长度与实际记录不一致")
curves["model"] = {"jaw_target": np.asarray([item["joint_targets"][-1] for item in steps]),
                   "jaw_actual": np.asarray([item["joint_position"][-1] for item in steps]),
                   "bilateral_force": np.asarray([min(item["jaw_forces"]) for item in steps]),
                   "object_height": np.asarray([item["object_position_root"][2] for item in steps]),
                   "return": np.asarray([item["reward"] for item in steps]).cumsum()}
for name, values in curves.items():
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("实际控制曲线包含无效数值")
    times = (np.arange(len(values["return"])) + 1) * source["control_dt"]
    label = "scripted controller" if name == "scripted" else "已保存模型"
    color = "#16a34a" if name == "scripted" else "#2563eb"
    axes[0, 0].plot(times, values["jaw_target"], color=color, label=label + " · target")
    axes[0, 0].plot(times, values["jaw_actual"], color=color, linestyle="--", alpha=.7, label=label + " · actual")
    axes[0, 1].plot(times, values["bilateral_force"], color=color, label=label)
    axes[1, 0].plot(times, values["object_height"], color=color, label=label)
    axes[1, 1].plot(times, values["return"], color=color, label=label)
axes[0, 0].set(title="夹爪控制目标与实际角度", ylabel="rad")
axes[0, 1].set(title="双侧接触力的较小值", ylabel="N")
axes[0, 1].axhline(.5, color="#64748b", linestyle=":", label="接触阈值")
axes[1, 0].set(title="物体实际高度", ylabel="robot root frame · 米")
axes[1, 1].set(title="实际累计 reward", ylabel="return")
for axis in axes.flat:
    axis.set_xlabel("控制时间 · 秒")
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
figure.suptitle(f"SO-101 Lift · 完全相同的初始状态\nscripted 成功：{source['environments'][0]['success']}；模型成功：{paired['success']}")
figure.tight_layout()
path = args.output / "paired_control.png"
figure.savefig(path, dpi=170)
plt.close(figure)
with Image.open(path) as image:
    image.verify()
report = {"status": "actual_paired_control_curves_verified", "initial_observation_error": paired["initial_observation_error"],
          "scripted_task_success": source["environments"][0]["success"], "model_task_success": paired["success"],
          "native_trace_sha256": source["trace_sha256"], "paired_report_sha256": digest(args.paired),
          "model_sha256": paired["model_sha256"], "source_sha256": digest(Path(__file__)),
          "figure_sha256": digest(path),
          "curves": {name: {key: value.tolist() for key, value in values.items()} for name, values in curves.items()}}
with (args.output / "report.json").open("x") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({key: value for key, value in report.items() if key != "curves"}, indent=2))
