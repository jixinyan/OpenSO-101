import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openso101.rl.config import digest


parser = argparse.ArgumentParser()
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--replay-report", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
native = json.loads((args.native / "report.json").read_text())
replay = json.loads(args.replay_report.read_text())
if native["task"] != replay["task"] or digest(args.native / "trajectory.hdf5") != native["trace_sha256"]:
    raise ValueError("实际回放诊断需要已校验的相同任务来源")
source_episode = Path(replay["source_episode"])
if digest(source_episode) != replay["source_sha256"]:
    raise ValueError("来源 episode SHA256 不一致")
replay_path = args.replay_report.with_suffix(".hdf5")
if digest(replay_path) != replay["trajectory_sha256"]:
    raise ValueError("回放 trajectory SHA256 不一致")
if replay["start_frame"] != 0 or replay["completed_frames"] != replay["stop_frame"]:
    raise ValueError("诊断需要从初始帧开始的完整回放")
records, errors = {}, {}
with h5py.File(args.native / "trajectory.hdf5") as source, h5py.File(replay_path) as repeated:
    selected = np.flatnonzero(source["active"][:, 0])
    if len(selected) != replay["completed_frames"]:
        raise ValueError("来源与实际回放步骤数量不一致")
    fields = ("joint_position", "joint_velocity", "object_position_root", "jaw_forces")
    original = {name: source[name][selected, 0] for name in fields}
    actual = {name: repeated[name][:] for name in fields}
    if any(not np.isfinite(value).all() for record in (original, actual) for value in record.values()):
        raise ValueError("实际轨迹需要有限数值")
    for name in fields:
        difference = actual[name] - original[name]
        errors[name] = {"rmse_per_component": np.sqrt(np.square(difference).mean(0)).tolist(),
                        "maximum_absolute_error": float(np.abs(difference).max()),
                        "initial_absolute_error": np.abs(difference[0]).tolist(),
                        "final_absolute_error": np.abs(difference[-1]).tolist()}
    for name, record in (("source", original), ("replay", actual)):
        contact = (record["jaw_forces"] >= .5).all(-1)
        indices = np.flatnonzero(contact)
        records[name] = {"bilateral_contact_steps": int(contact.sum()),
                         "first_bilateral_contact_frame": int(indices[0]) if len(indices) else None,
                         "maximum_object_height": float(record["object_position_root"][:, 2].max()),
                         "final_object_position_root": record["object_position_root"][-1].tolist(),
                         "final_jaw_angle": float(record["joint_position"][-1, -1])}
report = {"scope": "actual_source_and_isaac_replay_diagnostic", "records": records, "errors": errors,
          "source_task_success": replay["source_task_success"], "replay_task_success": replay["task_success_verified"],
          "source_episode_sha256": replay["source_sha256"], "source_trace_sha256": native["trace_sha256"],
          "replay_report_sha256": digest(args.replay_report), "replay_trace_sha256": digest(replay_path),
          "source_sha256": digest(Path(__file__))}
with args.output.open("x") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
