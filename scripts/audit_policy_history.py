import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from openso101.rl.bounded_policy import BoundedActorCritic
from openso101.rl.config import CheckpointMeta, digest
from openso101.rl.sequence_supervision import demonstration_windows, sequence_loss, sequence_predictions


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--length", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=1e-6)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(4)
meta = CheckpointMeta.read(args.checkpoint)
metadata = json.loads((args.run / "demonstrations.json").read_text())
if (digest(args.run / "demonstrations.pt") != metadata["dataset_sha256"]
        or metadata["task"] != meta.task_id or metadata["task_profile"] != meta.task_profile):
    raise ValueError("连续控制检查需要模型对应的实际示范数据")
model_sha = digest(args.checkpoint / meta.checkpoint)
dataset = torch.load(args.run / "demonstrations.pt", map_location="cpu", weights_only=True)
if any(not torch.isfinite(value).all() for value in dataset.values()):
    raise ValueError("实际示范包含无效数值")
observations = TensorDict({"policy": dataset["observations"]}, batch_size=[len(dataset["actions"])])
backend = json.loads((args.checkpoint / "backend.json").read_text())
policy_config = backend["policy"].copy()
if policy_config.pop("class_name") != "BoundedActorCritic":
    raise ValueError("连续控制检查需要 BoundedActorCritic")
policy = BoundedActorCritic(observations, backend["obs_groups"], 6, **policy_config)
policy.load_state_dict(torch.load(args.checkpoint / meta.checkpoint, map_location="cpu",
                                   weights_only=False)["model_state_dict"])
policy.eval()
windows, action_slice, dimensions = demonstration_windows(metadata, len(dataset["actions"]), args.length, "cpu")
if dimensions != dataset["observations"].shape[1]:
    raise ValueError("实际观测字段与数据尺寸不一致")
selected = windows[torch.linspace(0, len(windows) - 1, min(256, len(windows))).long()]
with torch.inference_mode():
    forced = policy.act_inference(observations)
    recurrent = torch.cat([sequence_predictions(policy, dataset["observations"], batch, action_slice)[1]
                           for batch in selected.split(32)])
    first_error = (recurrent[:, 0] - forced[selected[:, 0]]).abs().max().item()
    errors = (recurrent - dataset["actions"][selected]).square().mean(dim=(0, 2))
if first_error > 1e-6:
    raise RuntimeError("连续前向计算的首步与实际模型推理不一致")
gains = []
for index in torch.linspace(0, len(dataset["actions"]) - 1, 32).long():
    measured = dataset["observations"][index:index + 1].clone().requires_grad_()

    def predict(value):
        obs = TensorDict({"policy": value}, batch_size=[1])
        return policy.actor(policy.actor_obs_normalizer(policy.get_actor_obs(obs))).tanh()

    jacobian = torch.autograd.functional.jacobian(predict, measured)[0, :, 0, action_slice]
    gains.append(float(torch.linalg.svdvals(jacobian)[0]))
if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
    raise ValueError("连续监督检查需要正数有限学习率")
optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)
batch = selected[:32]
before, before_mse = sequence_loss(policy, dataset["observations"], dataset["actions"], batch, action_slice,
                                   margin=meta.config.demonstration_action_margin, objective="bounded_mse")
optimizer.zero_grad()
before.backward()
gradient_norm = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.)
if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
    raise RuntimeError("实际模型连续监督没有有效的 gradient")
optimizer.step()
if any(not torch.isfinite(value).all() for value in policy.parameters()):
    raise RuntimeError("连续监督更新产生无效参数")
with torch.inference_mode():
    after, after_mse = sequence_loss(policy, dataset["observations"], dataset["actions"], batch, action_slice,
                                    margin=meta.config.demonstration_action_margin, objective="bounded_mse")
args.output.mkdir(parents=True)
torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
           args.output / "gradient_check.pt")
if digest(args.checkpoint / meta.checkpoint) != model_sha:
    raise RuntimeError("来源模型的 SHA256 已改变")
report = {"status": "actual_policy_history_audited", "task": meta.task_id, "task_profile": meta.task_profile,
          "scope": "actual_recorded_states_with_model_previous_actions", "frames": len(dataset["actions"]),
          "windows": len(windows), "checked_windows": len(selected), "sequence_length": args.length,
          "single_step_action_mse": float((forced - dataset["actions"]).square().mean()),
          "sequence_mse_per_step": errors.tolist(), "first_step_maximum_error": first_error,
          "history_jacobian_maximum_gain": max(gains), "history_jacobian_median_gain": float(np.median(gains)),
          "gradient_norm": float(gradient_norm), "gradient_update_verified": True,
          "gradient_check_objective": "bounded_mse",
          "gradient_check_learning_rate": args.learning_rate,
          "gradient_check_objective_reduced": bool(after < before.detach()),
          "gradient_check_objective_before": float(before.detach()), "gradient_check_objective_after": float(after),
          "gradient_check_action_mse_before": float(before_mse.detach()), "gradient_check_action_mse_after": float(after_mse),
          "checkpoint_sha256": model_sha, "dataset_sha256": metadata["dataset_sha256"],
          "gradient_check_model_sha256": digest(args.output / "gradient_check.pt"),
          "source_sha256": {str(path): digest(path) for path in (Path(__file__),
              Path("src/openso101/rl/sequence_supervision.py"))},
          "gpu_tests_started": False, "closed_loop_task_success_verified": False}
with (args.output / "report.json").open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report, indent=2))
