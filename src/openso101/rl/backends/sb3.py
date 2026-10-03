# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import ClassVar

import torch
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

from openso101.rl.config import CheckpointMeta, TrainCfg


class Backend:
    algorithms: ClassVar = {"ppo": PPO, "sac": SAC, "tqc": TQC}

    def train(self, env, cfg: TrainCfg, output: Path, resume: Path | None = None) -> Path:
        wrapped = Sb3VecEnvWrapper(env, fast_variant=True)
        if resume:
            meta = CheckpointMeta.read(resume)
            wrapped = VecNormalize.load(str(resume / "normalization.pkl"), wrapped)
            model = self.algorithms[cfg.algo].load(str(resume / meta.checkpoint), env=wrapped)
            if cfg.algo != "ppo":
                model.load_replay_buffer(str(resume / "replay.pkl"))
        else:
            wrapped = VecNormalize(wrapped, norm_obs=cfg.normalize_observations, norm_reward=False, clip_obs=10.)
            common = {"learning_rate": cfg.learning_rate, "gamma": cfg.gamma, "seed": cfg.seed,
                      "policy_kwargs": {"net_arch": list(cfg.hidden_dims), "activation_fn": torch.nn.ELU},
                      "tensorboard_log": str(output), "device": env.unwrapped.device}
            if cfg.algo == "ppo":
                common.update(n_steps=cfg.rollout_steps, batch_size=cfg.batch_size(env.unwrapped.num_envs),
                              n_epochs=cfg.epochs, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip,
                              ent_coef=cfg.entropy_coef, max_grad_norm=cfg.max_grad_norm)
            else:
                common.update(buffer_size=cfg.replay_size, learning_starts=cfg.learning_starts,
                              batch_size=cfg.replay_batch_size, train_freq=1, gradient_steps=cfg.gradient_steps)
            model = self.algorithms[cfg.algo]("MlpPolicy", wrapped, **common)
        model.learn(total_timesteps=cfg.iterations * cfg.rollout_steps * env.unwrapped.num_envs,
                    reset_num_timesteps=resume is None)
        model.save(str(output / "model.zip"))
        wrapped.save(str(output / "normalization.pkl"))
        if cfg.algo != "ppo":
            model.save_replay_buffer(str(output / "replay.pkl"))
        return output / "model.zip"

    def load(self, env, folder: Path):
        meta = CheckpointMeta.read(folder)
        wrapped = VecNormalize.load(str(folder / "normalization.pkl"), Sb3VecEnvWrapper(env, fast_variant=True))
        wrapped.training = False
        wrapped.norm_reward = False
        model = self.algorithms[meta.config.algo].load(str(folder / meta.checkpoint), env=wrapped)

        def policy(observation):
            values = observation["policy"].detach().cpu().numpy()
            normalized = wrapped.normalize_obs(values)
            action, _ = model.predict(normalized, deterministic=True)
            return torch.as_tensor(action, device=env.unwrapped.device)

        return policy
