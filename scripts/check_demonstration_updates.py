import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from tensordict import TensorDict

from openso101.rl.bounded_policy import BoundedActorCritic
from openso101.rl.checked_ppo import CheckedPPO
from openso101.rl.config import CheckpointMeta, TrainCfg, digest
from openso101.rl.demonstrations import DemonstrationUpdates
from openso101.rl.stopping import TrainingStopRequest


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("连续监督程序检查需要禁止使用 CUDA")
meta = CheckpointMeta.read(args.run)
cfg = TrainCfg.model_validate_json(args.config.read_text())
torch.set_num_threads(4)
torch.manual_seed(cfg.seed)
source_sha = digest(args.run / meta.checkpoint)
metadata = json.loads((args.run / "demonstrations.json").read_text())
if digest(args.run / "demonstrations.pt") != metadata["dataset_sha256"] or metadata["task"] != meta.task_id:
    raise ValueError("连续监督程序检查需要实际模型与采集数据")
dataset = torch.load(args.run / "demonstrations.pt", map_location="cpu", weights_only=True)
obs = TensorDict({"policy": dataset["observations"]}, batch_size=[len(dataset["actions"])])
backend = json.loads((args.run / "backend.json").read_text())
policy_cfg = backend["policy"].copy()
algorithm_cfg = backend["algorithm"].copy()
if policy_cfg.pop("class_name") != "BoundedActorCritic" or algorithm_cfg.pop("class_name") != "PPO":
    raise ValueError("连续监督程序检查需要实际 bounded PPO 配置")
policy = BoundedActorCritic(obs, backend["obs_groups"], 6, **policy_cfg)
source_checkpoint = torch.load(args.run / meta.checkpoint, map_location="cpu", weights_only=False)
policy.load_state_dict(source_checkpoint["model_state_dict"])
policy.pin_normalization_reference()
args.output.mkdir(parents=True, exist_ok=False)
for name in ("demonstrations.json", "demonstrations.pt"):
    shutil.copy2(args.run / name, args.output / name)
algorithm = CheckedPPO(policy, diagnostic_dir=args.output, device="cpu", **algorithm_cfg)
updates = DemonstrationUpdates(algorithm, cfg, args.output, TrainingStopRequest())
updates.optimizer.load_state_dict(source_checkpoint["infos"]["demonstration_optimizer"])
updates.steps = source_checkpoint["infos"]["demonstration_gradient_steps"]
before_steps = updates.steps
with torch.inference_mode():
    before_mse = float((policy.act_inference(obs) - dataset["actions"]).square().mean())
before = {name: value.detach().clone() for name, value in policy.named_parameters()}
loss = updates.update()
with torch.inference_mode():
    after_mse = float((policy.act_inference(obs) - dataset["actions"]).square().mean())
policy.verify_normalization_reference()
changes = {name: float((value - before[name]).detach().abs().max()) for name, value in policy.named_parameters()}
if max(changes.values()) <= 0 or updates.last_sequence_action_mse is None:
    raise RuntimeError("实际连续监督更新没有改变模型参数")
if after_mse > 1e-4 or not torch.isfinite(torch.tensor(loss)):
    raise RuntimeError("实际监督更新未达到动作误差要求")
checkpoint = args.output / "model.pt"
torch.save({"model_state_dict": policy.state_dict(), "demonstration_optimizer": updates.optimizer.state_dict(),
            "demonstration_gradient_steps": updates.steps}, checkpoint)
saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
updates.optimizer.load_state_dict(saved["demonstration_optimizer"])
if digest(args.run / meta.checkpoint) != source_sha:
    raise RuntimeError("连续监督检查改变了来源模型")
report = {"status": "actual_demonstration_update_cpu_verified", "task": meta.task_id,
          "frames": len(dataset["actions"]), "sequence_windows": len(updates.sequence_windows),
          "sequence_length": cfg.demonstration_sequence_length, "sequence_weight": cfg.demonstration_sequence_weight,
          "sequence_objective": "bounded_mse", "gradient_steps": updates.steps - before_steps,
          "restored_gradient_steps": before_steps, "online_learning_rate": cfg.demonstration_online_learning_rate,
          "all_frames_action_mse_before": before_mse, "all_frames_action_mse_after": after_mse,
          "action_mse": loss, "last_sequence_action_mse": updates.last_sequence_action_mse,
          "maximum_parameter_change": max(changes.values()), "normalization_reference_verified": True,
          "source_checkpoint_sha256": source_sha, "dataset_sha256": metadata["dataset_sha256"],
          "checked_model_sha256": digest(checkpoint), "requested_config_sha256": digest(args.config),
          "source_sha256": {str(path): digest(path) for path in (Path(__file__),
              Path("src/openso101/rl/demonstrations.py"), Path("src/openso101/rl/sequence_supervision.py"))},
          "gpu_tests_started": False, "rl_transitions": 0, "closed_loop_task_success_verified": False}
(args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False))
