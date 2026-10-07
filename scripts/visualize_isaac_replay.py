import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--replay-report", type=Path, required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = json.loads((args.native / "report.json").read_text())
replay = json.loads(args.replay_report.read_text())
replay_trace = args.replay_report.with_suffix(".hdf5")
if (digest(args.native / "trajectory.hdf5") != native["trace_sha256"]
        or digest(replay_trace) != replay["trajectory_sha256"]
        or digest(Path(replay["source_episode"])) != replay["source_sha256"]
        or native["recorded_episode"]["sha256"] != replay["source_sha256"]
        or native["task"] != replay["task"] or replay["start_frame"] != 0
        or replay["completed_frames"] != native["environments"][0]["control_steps"]):
    raise ValueError("图表需要相同来源的完整实际采集与回放")
fields = ("joint_position", "object_position_root", "jaw_forces")
with h5py.File(args.native / "trajectory.hdf5") as source, h5py.File(replay_trace) as repeated:
    selected = np.flatnonzero(source["active"][:, 0])
    original = {name: source[name][selected, 0] for name in fields}
    actual = {name: repeated[name][:] for name in fields}
if any(actual[name].shape != original[name].shape for name in fields):
    raise ValueError("图表的实际采集与回放字段形状需要一致")
if any(not np.isfinite(value).all() for record in (original, actual) for value in record.values()):
    raise ValueError("实际控制曲线包含无效数值")
titles = ("物体实际高度", "双侧接触力的较小值", "夹爪实际角度", "六个关节的最大位置误差")
labels = ("来源采集", "HDF5 回放", "接触阈值", "控制时间 · 秒")
heading = "SO-101 PickPlace · 实际采集与来源布局回放"
caption = (f"来源环境：{replay['source_num_envs']}；回放环境：{replay['replay_num_envs']}；"
           f"完整帧数：{replay['completed_frames']}；任务成功：{replay['task_success_verified']}")
with TTFont(args.font) as font:
    available = font.getBestCmap()
    if any(ord(character) not in available for character in heading + caption + "".join(titles + labels)
           if not character.isspace()):
        raise ValueError("字体需要包含全部图表文字")
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
times = (np.arange(len(selected)) + 1) * native["control_dt"]
figure, axes = plt.subplots(2, 2, figsize=(12, 7))
for label, record, color, style in zip(labels, (original, actual), ("#16a34a", "#2563eb"), ("-", "--")):
    axes[0, 0].plot(times, record["object_position_root"][:, 2], label=label, color=color, linestyle=style)
    axes[0, 1].plot(times, record["jaw_forces"].min(-1), label=label, color=color, linestyle=style)
    axes[1, 0].plot(times, record["joint_position"][:, -1], label=label, color=color, linestyle=style)
joint_error = np.abs(actual["joint_position"] - original["joint_position"]).max(-1)
axes[1, 1].plot(times, joint_error, color="#2563eb", label=labels[1])
axes[0, 1].axhline(.5, color="#64748b", linestyle=":", label=labels[2])
for axis, title, unit in zip(axes.flat, titles, ("robot root frame · m", "N", "rad", "rad"), strict=True):
    axis.set(title=title, ylabel=unit, xlabel=labels[3])
    axis.grid(alpha=.2)
    axis.legend(fontsize=9)
figure.suptitle(heading)
figure.text(.5, .02, caption, ha="center")
figure.tight_layout(rect=(0, .06, 1, 1))
args.output.mkdir(parents=True, exist_ok=False)
path = args.output / "isaac_replay.png"
figure.savefig(path, dpi=170)
plt.close(figure)
with Image.open(path) as image:
    image.verify()
result = {"status": "actual_isaac_replay_figure_verified", "source_sha256": digest(Path(__file__)),
          "figure_sha256": digest(path), "source_episode_sha256": replay["source_sha256"],
          "source_trace_sha256": native["trace_sha256"], "replay_trace_sha256": replay["trajectory_sha256"],
          "replay_report_sha256": digest(args.replay_report), "frames": len(selected),
          "maximum_joint_position_error_rad": float(joint_error.max()),
          "task_success_verified": replay["task_success_verified"],
          "rl_policy_success_verified": False,
          "bilateral_contact_steps": {name: int((record["jaw_forces"] >= .5).all(-1).sum())
                                     for name, record in (("source", original), ("replay", actual))}}
with (args.output / "report.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
