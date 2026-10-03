import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.distributions import Normal

from openso101.rl.config import CheckpointMeta, digest


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--portable", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
meta = CheckpointMeta.read(args.run)
policy = json.loads((args.portable / "policy.json").read_text())
report = json.loads((args.portable / "validation.json").read_text())
if (meta.config.action_distribution != "tanh_gaussian"
        or policy["checkpoint_sha256"] != meta.files[meta.checkpoint]
        or digest(args.portable / "isaac_validation.hdf5") != report["trace_sha256"]):
    raise ValueError("探索检查需要同一模型与已校验的实际原生轨迹")
model = torch.load(args.run / meta.checkpoint, map_location="cpu", weights_only=False)
std = model["model_state_dict"]["log_std"].clamp(-5., 2.).exp()
mapping = next(item for item in policy["action_mapping"] if item["joint_name"] == "Jaw")
with h5py.File(args.portable / "isaac_validation.hdf5", "r") as trace:
    actions = torch.from_numpy(trace["raw_action"][:, :, mapping["action_index"]])
    targets = trace["joint_targets"][:, :, -1]
if not torch.isfinite(actions).all() or (actions.abs() >= 1).any():
    raise ValueError("探索检查需要有效的确定性 Tanh 动作")
distribution = Normal(actions.atanh(), std[mapping["action_index"]])
thresholds = []
for threshold in (.4, .2, .1):
    normalized = (threshold - mapping["offset"]) / mapping["scale"]
    probability = distribution.cdf(torch.tensor(normalized).atanh())
    thresholds.append({"jaw_target_below_rad": threshold,
                       "conditional_probability_mean": float(probability.mean()),
                       "conditional_probability_min": float(probability.min()),
                       "conditional_probability_max": float(probability.max())})
result = {"status": "conditional_action_distribution_measured", "checkpoint_sha256": meta.files[meta.checkpoint],
          "native_trace_sha256": report["trace_sha256"], "samples": int(actions.numel()),
          "latent_jaw_std": float(std[mapping["action_index"]]),
          "deterministic_jaw_target_min_rad": float(np.min(targets)),
          "deterministic_jaw_target_max_rad": float(np.max(targets)), "thresholds": thresholds,
          "probability_scope": "实际确定性轨迹所访问状态中的保存模型动作分布",
          "training_sample_frequency_verified": False, "task_success_verified": False,
          "source_sha256": digest(Path(__file__))}
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
