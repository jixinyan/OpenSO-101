# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from openso101.rl.config import CheckpointMeta, TrainCfg, write_backend_config


def configuration(cfg: TrainCfg, device: str):
    return {
        "seed": cfg.seed, "device": device, "num_steps_per_env": cfg.rollout_steps,
        "save_interval": max(1, cfg.iterations),
        "logger": "tensorboard", "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {"class_name": "ActorCritic", "init_noise_std": 0.5,
                   "actor_obs_normalization": cfg.normalize_observations,
                   "critic_obs_normalization": cfg.normalize_observations,
                   "actor_hidden_dims": list(cfg.hidden_dims), "critic_hidden_dims": list(cfg.hidden_dims),
                   "activation": "elu"},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": cfg.epochs,
                      "num_mini_batches": cfg.mini_batches, "learning_rate": cfg.learning_rate,
                      "schedule": "fixed", "gamma": cfg.gamma, "lam": cfg.gae_lambda,
                      "entropy_coef": cfg.entropy_coef, "clip_param": cfg.clip,
                      "max_grad_norm": cfg.max_grad_norm, "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True},
    }


class Backend:
    def train(self, env, cfg: TrainCfg, output: Path, resume: Path | None = None) -> Path:
        cfg.batch_size(env.unwrapped.num_envs)
        config = configuration(cfg, env.unwrapped.device)
        write_backend_config(output, config)
        runner = OnPolicyRunner(RslRlVecEnvWrapper(env), config, log_dir=str(output), device=env.unwrapped.device)
        if resume:
            runner.load(str(resume / CheckpointMeta.read(resume).checkpoint))
        runner.learn(num_learning_iterations=cfg.iterations, init_at_random_ep_len=True)
        runner.save(str(output / "model.pt"))
        return output / "model.pt"

    def load(self, env, folder: Path):
        meta = CheckpointMeta.read(folder)
        config = json.loads((folder / "backend.json").read_text())
        runner = OnPolicyRunner(RslRlVecEnvWrapper(env), config, log_dir=None, device=env.unwrapped.device)
        runner.load(str(folder / meta.checkpoint), load_optimizer=False)
        inference = runner.get_inference_policy(device=env.unwrapped.device)
        return inference
