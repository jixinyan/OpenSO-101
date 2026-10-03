import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from openso101.rl.backends.rsl_rl import configuration
from openso101.rl.bounded_policy import BoundedActorCritic
from openso101.rl.config import CheckpointMeta, TrainCfg, digest, write_backend_config


parser = argparse.ArgumentParser()
parser.add_argument("--demonstrations", type=Path, action="append", required=True)
parser.add_argument("--train-config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=.0003)
parser.add_argument("--initial-noise-std", type=float, default=.05)
args = parser.parse_args()
config = TrainCfg.model_validate_json(args.train_config.read_text())
config = TrainCfg.model_validate(config.model_dump() | {"initial_noise_std": args.initial_noise_std})
if config.backend != "rsl_rl" or config.algo != "ppo" or config.action_distribution != "tanh_gaussian":
    raise ValueError("示范初始化需要 RSL Tanh Gaussian PPO")
if args.epochs <= 0 or args.batch_size <= 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
    raise ValueError("示范训练需要有效的 epochs、batch size 和 learning rate")
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.manual_seed(config.seed)
sources, observations, actions, returns, groups = [], [], [], [], []
task = None
terms = None
mapping = None
episode_count = 0
for source in args.demonstrations:
    report = json.loads((source / "report.json").read_text())
    if (report["task_profile"] != "grasp_v4" or report["successes"] != report["episodes"]
            or report["state_action_sampling"] != "observation_before_action_with_transition_reward"
            or digest(source / "trajectory.hdf5") != report["trace_sha256"]):
        raise ValueError("初始化需要完整成功且经过来源核查的实际示范")
    if task is not None and (task, terms, mapping) != (
            report["task"], report["policy_observation_terms"], report["policy_action_mapping"]):
        raise ValueError("全部示范的任务、观测与动作转换必须一致")
    task, terms, mapping = report["task"], report["policy_observation_terms"], report["policy_action_mapping"]
    archive = args.output / "demonstrations" / f"source_{len(sources):03d}"
    shutil.copytree(source, archive)
    with h5py.File(archive / "trajectory.hdf5") as trajectory:
        for environment in range(report["episodes"]):
            selected = np.flatnonzero(trajectory["active"][:, environment])
            success = trajectory["success"][selected, environment].astype(bool)
            if not len(selected) or not success[-1] or success[:-1].any():
                raise ValueError("示范需要在首次任务成功步骤结束")
            reward = trajectory["weighted_reward"][selected, environment].sum(axis=-1)
            discounted = np.empty_like(reward)
            following = 0.
            for index in range(len(reward) - 1, -1, -1):
                following = float(reward[index]) + config.gamma * following
                discounted[index] = following
            observations.append(trajectory["policy_observation"][selected, environment])
            actions.append(trajectory["policy_action"][selected, environment])
            returns.append(discounted[:, None])
            groups.append(np.full(len(selected), episode_count, dtype=int))
            episode_count += 1
    sources.append({"report_sha256": digest(archive / "report.json"), "trace_sha256": digest(archive / "trajectory.hdf5"),
                    "relative_directory": archive.relative_to(args.output).as_posix(), "episodes": report["episodes"]})
if episode_count < 4:
    raise ValueError("需要至少四条实际成功示范，以完整 episode 划分训练和留出数据")
observation = torch.from_numpy(np.concatenate(observations)).float()
action = torch.from_numpy(np.concatenate(actions)).float()
return_value = torch.from_numpy(np.concatenate(returns)).float()
group = torch.from_numpy(np.concatenate(groups))
if observation.shape[1] != sum(term["size"] for term in terms) or action.shape != (len(observation), 6):
    raise ValueError("示范的实际观测或动作形状与来源定义不一致")
if not all(torch.isfinite(value).all() for value in (observation, action, return_value)) or (action.abs() > 1).any():
    raise ValueError("示范需要有限数值与有效的 bounded action")
training = group != episode_count - 1
held_out = ~training
sample = TensorDict({"policy": observation[training]}, batch_size=[int(training.sum())])
backend = configuration(config, "cpu")
policy_cfg = dict(backend["policy"])
policy_cfg.pop("class_name")
policy = BoundedActorCritic(sample, backend["obs_groups"], 6, **policy_cfg)
policy.update_normalization(sample)
policy.eval()
normalized_actor = policy.actor_obs_normalizer(observation).detach()
normalized_critic = policy.critic_obs_normalizer(observation).detach()
latent_target = action.clamp(-.999, .999).atanh()
optimizer = torch.optim.Adam([*policy.actor.parameters(), *policy.critic.parameters()], lr=args.learning_rate)
scales = torch.tensor([entry["scale"] for entry in mapping])
if [entry["action_index"] for entry in mapping] != list(range(6)):
    raise ValueError("初始化需要原生顺序的六个位置动作")
training_indices = training.nonzero(as_tuple=True)[0]
best_error = float("inf")
best_state = None
history = []
for epoch in range(args.epochs):
    permutation = training_indices[torch.randperm(len(training_indices))]
    for indices in permutation.split(args.batch_size):
        actor_loss = (policy.actor(normalized_actor[indices]) - latent_target[indices]).square().mean()
        critic_loss = (policy.critic(normalized_critic[indices]) - return_value[indices]).square().mean()
        loss = actor_loss + .01 * critic_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("示范训练的 loss 无效")
        optimizer.zero_grad()
        loss.backward()
        if not all(torch.isfinite(parameter.grad).all() for parameter in optimizer.param_groups[0]["params"]):
            raise FloatingPointError("示范训练的 gradient 无效")
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.)
        optimizer.step()
    with torch.inference_mode():
        error = (policy.actor(normalized_actor).tanh() - action) * scales
        critic_error = policy.critic(normalized_critic) - return_value
        record = {"epoch": epoch + 1, "training_target_rmse_rad": float(error[training].square().mean().sqrt()),
                  "held_out_target_rmse_rad": float(error[held_out].square().mean().sqrt()),
                  "held_out_maximum_target_error_rad": float(error[held_out].abs().max()),
                  "held_out_critic_rmse": float(critic_error[held_out].square().mean().sqrt())}
    history.append(record)
    if record["held_out_target_rmse_rad"] < best_error:
        best_error = record["held_out_target_rmse_rad"]
        best_state = copy.deepcopy(policy.state_dict())
        best_epoch = epoch + 1
    if (epoch + 1) % 100 == 0:
        print(json.dumps(record), flush=True)
if best_state is None or not all(torch.isfinite(value).all() for value in best_state.values()):
    raise FloatingPointError("示范训练没有有效模型")
policy.load_state_dict(best_state)
ppo_optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
checkpoint = args.output / "model.pt"
torch.save({"model_state_dict": policy.state_dict(), "optimizer_state_dict": ppo_optimizer.state_dict(),
            "iter": -1, "infos": {"initialization": "actual_demonstration_behavior_cloning", "best_epoch": best_epoch}}, checkpoint)
write_backend_config(args.output, backend)
(args.output / "train.json").write_text(config.model_dump_json(indent=2))
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
package = Path(__file__).resolve().parents[1] / "src/openso101"
with ZipFile(args.output / "source.zip", "w") as archive:
    for source in sorted(package.rglob("*.py")):
        archive.write(source, source.relative_to(package.parent))
    archive.write(Path(__file__), "scripts/bootstrap_grasp_policy.py")
result = {"status": "demonstration_initialization_completed", "training_kind": "behavior_cloning",
          "task": task, "task_profile": "grasp_v4", "source_git_sha": revision, "source_sha256": digest(Path(__file__)),
          "episodes": episode_count, "actual_demonstration_transitions": len(observation), "completed_rl_transitions": 0,
          "training_samples": int(training.sum()), "held_out_samples": int(held_out.sum()),
          "held_out_episode": episode_count - 1, "best_epoch": best_epoch, "best_held_out_target_rmse_rad": best_error,
          "initial_noise_std": config.initial_noise_std, "observation_terms": terms, "action_mapping": mapping,
          "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
          "gamma": config.gamma, "sources": sources, "history": history,
          "checkpoint_sha256": digest(checkpoint), "rl_policy_success_verified": False}
(args.output / "bootstrap.json").write_text(json.dumps(result, indent=2) + "\n")
files = {path.relative_to(args.output).as_posix(): digest(path) for path in args.output.rglob("*") if path.is_file()}
CheckpointMeta(task_id=task, task_profile="grasp_v4", config=config, git_sha=revision, checkpoint=checkpoint.name,
               files=files, completed_transitions=0).write(args.output)
CheckpointMeta.read(args.output)
print(json.dumps({name: value for name, value in result.items() if name != "history"}), flush=True)
