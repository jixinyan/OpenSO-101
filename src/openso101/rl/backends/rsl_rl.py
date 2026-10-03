# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

from openso101.rl.config import CheckpointMeta, TrainCfg, write_backend_config
from openso101.rl.bounded_policy import runner_class
from openso101.rl.initialization import record_initial_std


def configuration(cfg: TrainCfg, device: str):
    return {
        "seed": cfg.seed, "device": device, "num_steps_per_env": cfg.rollout_steps,
        "action_likelihood": "gaussian_latent" if cfg.action_distribution == "tanh_gaussian" else "gaussian",
        "save_interval": min(50, cfg.iterations),
        "logger": "tensorboard", "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {"class_name": "BoundedActorCritic" if cfg.action_distribution == "tanh_gaussian" else "ActorCritic",
                   "init_noise_std": cfg.initial_noise_std,
                   "noise_std_type": "log",
                   "actor_obs_normalization": cfg.normalize_observations,
                   "critic_obs_normalization": cfg.normalize_observations,
                   "actor_hidden_dims": list(cfg.hidden_dims), "critic_hidden_dims": list(cfg.hidden_dims),
                   "activation": "elu"},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": cfg.epochs,
                      "num_mini_batches": cfg.mini_batches, "learning_rate": cfg.learning_rate,
                      "schedule": cfg.learning_rate_schedule, "desired_kl": cfg.desired_kl,
                      "gamma": cfg.gamma, "lam": cfg.gae_lambda,
                      "entropy_coef": cfg.entropy_coef, "clip_param": cfg.clip,
                      "max_grad_norm": cfg.max_grad_norm, "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True},
    }


class Backend:
    def train(self, env, cfg: TrainCfg, output: Path, resume: Path | None = None) -> Path:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        cfg.batch_size(env.unwrapped.num_envs)
        config = configuration(cfg, env.unwrapped.device)
        config["diagnostic_dir"] = str(output.resolve())
        config["save_interval"] = 1
        from openso101.tasks.shared.position_action import NormalizedJointPositionAction

        if (cfg.action_distribution == "tanh_gaussian"
                and isinstance(env.unwrapped.action_manager.get_term("arm_action"), NormalizedJointPositionAction)):
            from openso101.rl.vision_distillation import action_mapping
            from openso101.robots import SO101_ARM_JOINT_NAMES

            robot = env.unwrapped.scene["robot"]
            initial = [0.] * env.unwrapped.action_manager.total_action_dim
            for item in action_mapping(env.unwrapped):
                if item["joint_name"] not in SO101_ARM_JOINT_NAMES:
                    continue
                position = robot.data.default_joint_pos[0, robot.joint_names.index(item["joint_name"])]
                initial[item["action_index"]] = float(((position - item["offset"]) / item["scale"]).clamp(-.98, .98))
            config["policy"]["initial_action_mean"] = initial
        if resume:
            previous = CheckpointMeta.read(resume)
            if (cfg.hidden_dims, cfg.normalize_observations, cfg.action_distribution, cfg.environment_mode) != (
                previous.config.hidden_dims, previous.config.normalize_observations,
                previous.config.action_distribution, previous.config.environment_mode,
            ):
                raise ValueError("继续训练需要保持 policy 结构、观测归一化、动作分布和环境设置")
            config["policy"] = json.loads((resume / "backend.json").read_text())["policy"]
        write_backend_config(output, config)
        runner = runner_class(config)(RslRlVecEnvWrapper(env), config, log_dir=str(output), device=env.unwrapped.device)
        if resume:
            runner.load(str(resume / CheckpointMeta.read(resume).checkpoint))
            runner.current_learning_iteration += 1
        std = runner.alg.policy.log_std
        if cfg.action_distribution == "tanh_gaussian":
            std = std.clamp(-5., 2.)
        record_initial_std(output, cfg, std.exp(), resumed=resume is not None)
        from openso101.rl.benchmark import evaluate_snapshot

        remaining = cfg.iterations
        while remaining:
            count = min(remaining, cfg.evaluation_interval)
            try:
                runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
            except Exception as error:
                if hasattr(runner.alg, "capture_failure"):
                    runner.alg.capture_failure(f"{type(error).__name__}: {error}")
                raise
            checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
            runner.save(str(checkpoint))
            converged = evaluate_snapshot(output, checkpoint, runner.current_learning_iteration, cfg)
            remaining -= count
            if converged:
                break
            if remaining:
                runner.current_learning_iteration += 1
        runner.save(str(output / "model.pt"))
        return output / "model.pt"

    def load(self, env, folder: Path):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        meta = CheckpointMeta.read(folder)
        config = json.loads((folder / "backend.json").read_text())
        runner = runner_class(config)(RslRlVecEnvWrapper(env), config, log_dir=None, device=env.unwrapped.device)
        runner.load(str(folder / meta.checkpoint), load_optimizer=False)
        inference = runner.get_inference_policy(device=env.unwrapped.device)
        return inference
