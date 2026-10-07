import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from openso101.rl.config import digest
from openso101.rl.snapshot import TrainingRunMeta


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
metadata = TrainingRunMeta.read(args.run)
event_files = list(args.run.glob("events.out.tfevents.*"))
if len(event_files) != 1 or metadata.config.backend != "rsl_rl":
    raise ValueError("曲线需要一份实际 RSL TensorBoard 事件文件")
args.output.mkdir(parents=True, exist_ok=False)
event_copy = args.output / event_files[0].name
shutil.copy2(event_files[0], event_copy)
events = EventAccumulator(str(event_copy), size_guidance={"scalars": 0}).Reload()
tags = {
    "Train/mean_reward": ("平均 episode return", "return"),
    "Episode_Reward/progress": ("任务进展 reward", "日志记录的数值"),
    "Metrics/object_pose/position_error": ("物体到目标的距离", "米"),
    "Loss/value_function": ("Value function loss", "loss"),
    "Policy/mean_noise_std": ("动作探索标准差", "Gaussian 标准差"),
    "Episode_Termination/success": ("训练中的任务成功比例", "成功比例"),
}
curves = {}
for name in tags:
    entries = events.Scalars(name)
    iterations = np.asarray([entry.step for entry in entries], dtype=int)
    values = np.asarray([entry.value for entry in entries])
    if (not len(entries) or not np.isfinite(values).all() or (np.diff(iterations) <= 0).any()
            or iterations[0] != metadata.start_iteration):
        raise ValueError("实际事件数值或起始 iteration 与训练 metadata 不一致")
    curves[name] = [{"iteration": int(entry.step), "value": float(entry.value),
                     "completed_transitions": metadata.prior_transitions +
                     (int(entry.step) - metadata.start_iteration + 1) *
                     metadata.config.rollout_steps * metadata.num_envs}
                    for entry in entries]
font_manager.fontManager.addfont(args.font)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for axis, (name, (title, ylabel)) in zip(axes.flat, tags.items(), strict=True):
    curve = curves[name]
    axis.plot([item["completed_transitions"] / 1e6 for item in curve],
              [item["value"] for item in curve], color="#2563eb", linewidth=1.5)
    axis.set(title=title, xlabel="实际累计 transitions（百万）", ylabel=ylabel)
    axis.grid(alpha=.2)
figure.suptitle(f"SO-101 Lift：seed {metadata.config.seed}，{metadata.num_envs} 个环境\n实际 TensorBoard 记录", fontsize=15)
figure.tight_layout()
path = args.output / "continued_training.png"
figure.savefig(path, dpi=160)
plt.close(figure)
with Image.open(path) as image:
    image.verify()
with Image.open(path) as image:
    dimensions = [image.width, image.height]
report = {"status": "actual_training_event_snapshot_verified", "created_at": datetime.now(UTC).isoformat(),
          "training_git_sha": metadata.git_sha, "run_metadata_sha256": digest(args.run / "run.json"),
          "events_sha256": digest(event_copy), "source_sha256": digest(Path(__file__)),
          "prior_transitions": metadata.prior_transitions, "start_iteration": metadata.start_iteration,
          "num_envs": metadata.num_envs, "curves": curves,
          "figure_sha256": digest(path), "figure_dimensions": dimensions,
          "independent_task_success_verified": False}
(args.output / "training_snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "latest": {name: values[-1] for name, values in curves.items()}}, indent=2))
