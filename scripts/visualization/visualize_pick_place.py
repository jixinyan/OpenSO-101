import argparse
import json
from pathlib import Path

import av
import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = json.loads((args.source / "report.json").read_text())
states = json.loads((args.source / "initial_states.json").read_text())
if report["task"] != "OpenSO101-PickPlace-v0":
    raise ValueError("需要原生 PickPlace 轨迹")
for filename, field in (("trajectory.hdf5", "trace_sha256"), ("initial_states.json", "states_sha256")):
    if digest(args.source / filename) != report[field]:
        raise ValueError(f"源文件的 SHA256 不一致：{filename}")
args.output.mkdir(parents=True, exist_ok=False)
records = []
fig, axes = plt.subplots(report["episodes"], 4, figsize=(17, 3.5 * report["episodes"]), squeeze=False)
with h5py.File(args.source / "trajectory.hdf5") as trace:
    for index, environment in enumerate(report["environments"]):
        selected = np.flatnonzero(trace["active"][:, index])
        positions = trace["object_position_root"][selected, index]
        goals = np.asarray(states["environments"][index]["place_goal_position_root"])
        distances = np.linalg.norm(positions - goals, axis=-1)
        phases = trace["phase"][selected, index]
        holds = trace["placement_hold_seconds"][selected, index]
        angles = trace["joint_position"][selected, index, -1]
        times = (selected + 1) * report["control_dt"]
        actual_success = trace["success"][selected, index].astype(bool)
        if bool(actual_success.any()) != environment["success"]:
            raise ValueError("逐步骤任务成功与源报告不一致")
        if not all(np.isfinite(value).all() for value in (positions, distances, holds, angles)):
            raise ValueError("原生轨迹包含无效数值")
        axes[index, 0].plot(times, positions[:, 2], label="Object height")
        axes[index, 0].axhline(goals[2], linestyle="--", label="Final goal height")
        axes[index, 0].set(ylabel=f"Env {index} height (m)")
        axes[index, 1].plot(times, distances, label="Distance to final goal")
        axes[index, 1].set(ylabel="Distance (m)")
        axes[index, 2].plot(times, holds, label="Actual placement hold")
        axes[index, 2].set(ylabel="Placement hold (s)")
        axes[index, 3].plot(times, phases, label="Executed phase")
        axes[index, 3].set(yticks=np.arange(-1, 7), yticklabels=("Settle", "Approach", "Grasp", "Close", "Lift", "Carry", "Place", "Release"))
        for axis in axes[index]:
            axis.set_xlabel("Actual episode time (s)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        records.append({"environment": index, "success": environment["success"],
                        "control_steps": int(len(selected)), "final_goal_distance_m": float(distances[-1]),
                        "final_placement_hold_seconds": float(holds[-1]), "final_jaw_angle_rad": float(angles[-1]),
                        "phase_steps": {str(int(phase)): int((phases == phase).sum()) for phase in np.unique(phases)}})
fig.suptitle(f"Actual Isaac PickPlace · scripted IK · {report['successes']}/{report['episodes']} successful episodes\n"
             "Task thresholds, physical limits and eight-second episode budget preserved")
fig.tight_layout()
fig.savefig(args.output / "native_pick_place.png", dpi=170)
plt.close(fig)
video = None
if "video" in report:
    source_video = args.source / Path(report["video"]["path"]).name
    if digest(source_video) != report["video"]["sha256"]:
        raise ValueError("原生双相机视频的 SHA256 不一致")
    with av.open(str(source_video)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        frames = 0
        for frame in container.decode(stream):
            if frame.width != 512 or frame.height != 256:
                raise ValueError("实际双相机视频的尺寸与源报告不一致")
            frames += 1
    if frames != report["video"]["frames"] or rate != report["video"]["fps"]:
        raise ValueError("实际解码的视频尺寸、帧数或帧率与源报告不一致")
    video = {"sha256": digest(source_video), "decoded_frames": frames, "fps": rate,
             "camera_order": report["video"]["camera_order"]}
result = {"status": "actual_pick_place_visualization_recorded", "environments": records,
          "native_report_sha256": digest(args.source / "report.json"), "trace_sha256": report["trace_sha256"],
          "source_sha256": digest(Path(__file__)), "figure_sha256": digest(args.output / "native_pick_place.png"),
          "video": video, "rl_policy_success_verified": False}
(args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
