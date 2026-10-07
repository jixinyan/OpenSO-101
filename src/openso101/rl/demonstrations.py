import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from .config import CheckpointMeta, digest


def prepare_demonstrations(env, cfg, output, task_id, task_profile, resume=None):
    if not cfg.demonstration_sources:
        return {}
    names = ("demonstrations.pt", "demonstrations.json")
    if resume is not None:
        previous = CheckpointMeta.read(resume)
        if previous.config.demonstration_sources != cfg.demonstration_sources:
            raise ValueError("继续训练需要保持成功示范来源")
        if previous.config.demonstration_include_failed_supervision != cfg.demonstration_include_failed_supervision:
            raise ValueError("继续训练需要保持已校验的动作监督范围")
        for name in names:
            if name not in previous.files:
                raise ValueError("继续训练的 checkpoint 缺少已校验示范文件")
            shutil.copy2(resume / name, output / name)
        return {name: digest(output / name) for name in names}
    from .vision_distillation import action_mapping

    runtime = env.unwrapped
    expected_terms = [{"name": name, "size": int(np.prod(shape))} for name, shape in zip(
        runtime.observation_manager.active_terms["policy"],
        runtime.observation_manager.group_obs_term_dim["policy"], strict=True)]
    expected_mapping = action_mapping(runtime)
    buffers = {name: [] for name in ("observations", "actions", "returns")}
    sources = []
    for source in cfg.demonstration_sources:
        folder = Path(source).resolve()
        report = json.loads((folder / "report.json").read_text())
        if (report["task"] != task_id or report["task_profile"] != task_profile
                or report["environment_mode"] != cfg.environment_mode
                or report["state_action_sampling"] != "observation_before_action_with_transition_reward"
                or report["policy_observation_terms"] != expected_terms
                or report["policy_action_mapping"] != expected_mapping
                or report["reward_terms"] != list(runtime.reward_manager.active_terms)
                or report["profile_sha256"] != digest(Path(f"src/openso101/tasks/shared/{task_profile}.py"))
                or report["control_dt"] != runtime.step_dt):
            raise ValueError("成功示范的任务、观测、动作或控制配置与训练不一致")
        if digest(folder / "trajectory.hdf5") != report["trace_sha256"]:
            raise ValueError("成功示范 trajectory SHA256 不一致")
        action_field = "policy_action"
        if "supervision" in report:
            supervision = report["supervision"]
            if (supervision["action_field"] != "expert_policy_action"
                    or supervision["executed_action_field"] != "policy_action"
                    or supervision["reward_source"] != "actual_executed_transition"):
                raise ValueError("示范监督目标与实际动作的来源不符合支持的定义")
            action_field = supervision["action_field"]
        with h5py.File(folder / "trajectory.hdf5", "r") as stream:
            arrays = {name: stream[name][:] for name in
                      ("policy_observation", "policy_action", "weighted_reward", "active", "success", "terminated", "truncated")}
            if action_field != "policy_action":
                arrays[action_field] = stream[action_field][:]
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("成功示范包含无效数值")
        included = []
        for episode in report["environments"]:
            failed_supervision = (not episode["success"] and cfg.demonstration_include_failed_supervision
                                  and action_field == "expert_policy_action")
            if not episode["success"] and not failed_supervision:
                continue
            index = episode["environment"]
            selected = np.flatnonzero(arrays["active"][:, index])
            if (len(selected) != episode["control_steps"] or selected[0] != 0
                    or not np.array_equal(selected, np.arange(len(selected)))
                    or bool(arrays["success"][selected[-1], index]) != episode["success"]
                    or not bool((arrays["terminated"] | arrays["truncated"])[selected[-1], index])):
                raise ValueError("动作监督需要从初始状态到实际终止的完整 episode")
            observations = arrays["policy_observation"][selected, index]
            actions = arrays[action_field][selected, index]
            rewards = arrays["weighted_reward"][selected, index].sum(axis=-1)
            if observations.shape != (len(selected), sum(item["size"] for item in expected_terms)):
                raise ValueError("示范观测尺寸与训练不一致")
            if actions.shape != (len(selected), runtime.action_manager.total_action_dim) or np.abs(actions).max() > 1:
                raise ValueError("示范动作需要有效的 bounded joint targets")
            returns = np.empty_like(rewards)
            value = 0.
            for step in range(len(rewards) - 1, -1, -1):
                value = float(rewards[step]) + cfg.gamma * value
                returns[step] = value
            for name, array in (("observations", observations), ("actions", actions), ("returns", returns)):
                buffers[name].append(torch.from_numpy(array.copy()).float())
            included.append({"environment": index, "frames": len(selected), "return": float(rewards.sum()),
                             "success": episode["success"], "failed_expert_supervision": failed_supervision})
        if not included:
            raise ValueError("示范来源没有实际成功的完整 episode")
        sources.append({"source": str(folder), "report_sha256": digest(folder / "report.json"),
                        "trajectory_sha256": report["trace_sha256"], "episodes": included,
                        "supervision_action_field": action_field, "executed_action_field": "policy_action",
                        "reward_source": "actual_executed_transition",
                        "controller": report["controller"]})
    dataset = {name: torch.cat(items) for name, items in buffers.items()}
    torch.save(dataset, output / names[0])
    (output / names[1]).write_text(json.dumps({
        "schema_version": 1, "task": task_id, "task_profile": task_profile,
        "sources": sources, "frames": len(dataset["actions"]), "gamma": cfg.gamma,
        "successful_episodes": sum(item["success"] for source in sources for item in source["episodes"]),
        "failed_expert_supervision_episodes": sum(item["failed_expert_supervision"] for source in sources for item in source["episodes"]),
        "state_action_sampling": "observation_before_action_with_transition_reward",
        "policy_observation_terms": expected_terms, "policy_action_mapping": expected_mapping,
        "dataset_sha256": digest(output / names[0]),
    }, indent=2) + "\n")
    return {name: digest(output / name) for name in names}


class DemonstrationUpdates:
    def __init__(self, algorithm, cfg, folder, stop_request):
        self.algorithm, self.cfg = algorithm, cfg
        self.stop_request = stop_request
        metadata = json.loads((folder / "demonstrations.json").read_text())
        if digest(folder / "demonstrations.pt") != metadata["dataset_sha256"]:
            raise ValueError("成功示范 dataset SHA256 不一致")
        self.dataset = {name: value.to(algorithm.device) for name, value in torch.load(
            folder / "demonstrations.pt", weights_only=True).items()}
        self.optimizer = torch.optim.Adam([
            *algorithm.policy.actor.parameters(), *algorithm.policy.critic.parameters()],
            lr=cfg.demonstration_learning_rate)
        self.steps = 0
        self.latent_targets = self.dataset["actions"].clamp(
            -1 + cfg.demonstration_action_margin, 1 - cfg.demonstration_action_margin).atanh()

    def losses(self, indices):
        observations = TensorDict({"policy": self.dataset["observations"][indices]}, batch_size=[len(indices)])
        policy = self.algorithm.policy
        mean = policy.actor(policy.actor_obs_normalizer(policy.get_actor_obs(observations)))
        actions = mean.tanh()
        values = policy.evaluate(observations).squeeze(-1)
        action_mse = torch.nn.functional.mse_loss(actions, self.dataset["actions"][indices])
        actor = (torch.nn.functional.mse_loss(mean, self.latent_targets[indices])
                 if self.cfg.demonstration_objective == "latent_mse" else action_mse)
        critic = torch.nn.functional.smooth_l1_loss(values, self.dataset["returns"][indices])
        return actor, critic, action_mse

    def step(self, indices):
        self.stop_request.check()
        actor, critic, action_mse = self.losses(indices)
        loss = actor + .01 * critic
        self.algorithm.require_finite("demonstration_loss", loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.algorithm.policy.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        for name, parameter in self.algorithm.policy.named_parameters():
            self.algorithm.require_finite(f"demonstration_parameter/{name}", parameter)
        self.steps += 1
        return float(actor.detach()), float(critic.detach()), float(action_mse.detach())

    def pretrain(self, folder):
        policy = self.algorithm.policy
        observations = TensorDict({"policy": self.dataset["observations"]},
                                  batch_size=[len(self.dataset["actions"])])
        policy.train()
        if self.cfg.freeze_demonstration_normalization:
            policy.initialize_normalization_reference(observations)
        else:
            policy.update_normalization(observations)
        count = len(self.dataset["actions"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.demonstration_epochs,
            eta_min=self.cfg.demonstration_learning_rate * .01)
        with (folder / "demonstration_pretrain.jsonl").open("x") as stream:
            for epoch in range(self.cfg.demonstration_epochs):
                permutation = torch.randperm(count, device=self.algorithm.device)
                losses = [self.step(indices) for indices in permutation.split(self.cfg.demonstration_batch_size)]
                scheduler.step()
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    record = {"epoch": epoch + 1, "actor_mse": float(np.mean([item[2] for item in losses])),
                              "objective_mse": float(np.mean([item[0] for item in losses])),
                              "objective": self.cfg.demonstration_objective,
                              "learning_rate": scheduler.get_last_lr()[0],
                              "critic_smooth_l1": float(np.mean([item[1] for item in losses])),
                              "gradient_steps": self.steps}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    print(json.dumps({"demonstration_pretrain": record}), flush=True)
        with torch.no_grad():
            actor, critic, action_mse = self.losses(torch.arange(count, device=self.algorithm.device))
            predictions = policy.act_inference(observations)
            joint_rmse = (predictions - self.dataset["actions"]).square().mean(0).sqrt()
        report = {"status": "actual_success_demonstration_initialization_completed", "frames": count,
                  "actor_mse": float(action_mse), "objective_mse": float(actor),
                  "objective": self.cfg.demonstration_objective,
                  "action_margin": self.cfg.demonstration_action_margin,
                  "rmse_per_action": joint_rmse.tolist(), "critic_smooth_l1": float(critic),
                  "gradient_steps": self.steps, "rl_transitions": 0,
                  "dataset_sha256": digest(folder / "demonstrations.pt"), "task_success_verified": False}
        (folder / "demonstration_initialization.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def update(self):
        losses = []
        for _ in range(self.cfg.demonstration_updates_per_iteration):
            indices = torch.randint(len(self.dataset["actions"]), (self.cfg.demonstration_batch_size,),
                                    device=self.algorithm.device)
            losses.append(self.step(indices)[2])
        return float(np.mean(losses))
