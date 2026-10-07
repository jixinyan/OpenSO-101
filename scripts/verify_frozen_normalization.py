import argparse
import json
from pathlib import Path

import torch

from openso101.rl.config import CheckpointMeta, digest


parser = argparse.ArgumentParser()
parser.add_argument("--initial", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
initial_meta = CheckpointMeta.read(args.initial)
run_meta = CheckpointMeta.read(args.run)
if not run_meta.config.freeze_demonstration_normalization or not run_meta.config.normalize_observations:
    raise ValueError("检查需要固定观测归一化的实际训练")
if (initial_meta.task_id, initial_meta.task_profile) != (run_meta.task_id, run_meta.task_profile):
    raise ValueError("初始模型与训练任务配置需要一致")
reference_path = args.initial / initial_meta.checkpoint
reference = torch.load(reference_path, map_location="cpu", weights_only=False)["model_state_dict"]
fields = [name for name in reference if name.startswith(("actor_obs_normalizer.", "critic_obs_normalizer."))]
if len(fields) != 8 or any(not torch.isfinite(reference[name]).all() for name in fields):
    raise ValueError("初始模型需要完整且有效的归一化统计")
counts = {name: int(reference[f"{name}_obs_normalizer.count"]) for name in ("actor", "critic")}
if min(counts.values()) <= 0:
    raise ValueError("归一化统计需要实际样本")
paths = sorted(args.run.glob("model_*.pt")) + [args.run / run_meta.checkpoint]
if not paths:
    raise ValueError("需要实际保存的训练模型")
records = []
for path in dict.fromkeys(paths):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["infos"]["normalization_reference_verified"] is not True:
        raise ValueError("保存模型缺少归一化检查记录")
    state = checkpoint["model_state_dict"]
    if any(not torch.equal(state[name], reference[name]) for name in fields):
        raise RuntimeError("保存模型的归一化统计与初始模型不一致")
    records.append({"file": path.name, "iteration": checkpoint["iter"], "sha256": digest(path)})
result = {"status": "all_saved_normalization_statistics_verified", "models": records,
          "counts": counts, "fields": fields, "initial_checkpoint_sha256": digest(reference_path),
          "source_sha256": digest(Path(__file__)), "task_success_verified": False}
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
