import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from openso101.rl.bounded_policy import BoundedActorCritic
from openso101.rl.config import CheckpointMeta, digest


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(4)
meta = CheckpointMeta.read(args.checkpoint)
demonstrations = json.loads((args.run / "demonstrations.json").read_text())
if (digest(args.run / "demonstrations.pt") != demonstrations["dataset_sha256"]
        or demonstrations["task"] != meta.task_id or demonstrations["task_profile"] != meta.task_profile):
    raise ValueError("检查需要模型对应的已校验实际示范")
dataset = torch.load(args.run / "demonstrations.pt", map_location="cpu", weights_only=True)
observations = TensorDict({"policy": dataset["observations"]}, batch_size=[len(dataset["actions"])])
backend = json.loads((args.checkpoint / "backend.json").read_text())
policy_config = backend["policy"].copy()
if policy_config.pop("class_name") != "BoundedActorCritic":
    raise ValueError("检查需要 BoundedActorCritic")
policy = BoundedActorCritic(observations, backend["obs_groups"], dataset["actions"].shape[1], **policy_config)
policy.load_state_dict(torch.load(args.checkpoint / meta.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
policy.eval()
offset = 0
action_slice = None
for term in demonstrations["policy_observation_terms"]:
    if term["name"] == "actions":
        action_slice = slice(offset, offset + term["size"])
    offset += term["size"]
if action_slice is None or offset != dataset["observations"].shape[1]:
    raise ValueError("实际观测需要完整记录 previous actions")
with torch.inference_mode():
    predicted = policy.act_inference(observations).numpy()
previous = dataset["observations"][:, action_slice].numpy()
target = dataset["actions"].numpy()
if predicted.shape != target.shape or previous.shape != target.shape:
    raise ValueError("模型、标签和 previous actions 尺寸不一致")
if any(not np.isfinite(value).all() for value in (predicted, target, previous)):
    raise ValueError("实际模型检查产生无效数值")
target_delta, predicted_delta = target - previous, predicted - previous
moving = np.abs(target_delta[:, :5]) >= .001
records = []
offset = 0
for source in demonstrations["sources"]:
    for episode in source["episodes"]:
        selected = slice(offset, offset + episode["frames"])
        records.append({"source": source["source"], "environment": episode["environment"],
                        "frames": episode["frames"],
                        "model_rmse_per_action": np.sqrt(((predicted[selected] - target[selected]) ** 2).mean(0)).tolist(),
                        "previous_action_rmse_per_action": np.sqrt((target_delta[selected] ** 2).mean(0)).tolist(),
                        "first_previous_action": previous[offset].tolist(),
                        "first_target_action": target[offset].tolist(), "first_model_action": predicted[offset].tolist()})
        offset += episode["frames"]
if offset != len(target) or not moving.any():
    raise ValueError("实际示范的 episode 范围或运动样本无效")
result = {"status": "actual_demonstration_action_motion_audited", "frames": len(target),
          "model_mse": float(((predicted - target) ** 2).mean()),
          "previous_action_mse": float((target_delta ** 2).mean()),
          "arm_model_mse": float(((predicted[:, :5] - target[:, :5]) ** 2).mean()),
          "arm_previous_action_mse": float((target_delta[:, :5] ** 2).mean()),
          "arm_motion_components": int(moving.sum()),
          "arm_motion_direction_agreement": float((np.sign(predicted_delta[:, :5][moving])
                                                    == np.sign(target_delta[:, :5][moving])).mean()),
          "model_rmse_per_action": np.sqrt(((predicted - target) ** 2).mean(0)).tolist(),
          "previous_action_rmse_per_action": np.sqrt((target_delta ** 2).mean(0)).tolist(),
          "episodes": records, "dataset_sha256": digest(args.run / "demonstrations.pt"),
          "checkpoint_sha256": digest(args.checkpoint / meta.checkpoint), "source_sha256": digest(Path(__file__)),
          "closed_loop_task_success_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))
