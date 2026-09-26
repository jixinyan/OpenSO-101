# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils import set_seed
from skrl.utils.runner.torch import Runner

from openso101.rl.config import CheckpointMeta, TrainCfg, write_backend_config


def configuration(cfg: TrainCfg, output: Path):
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": list(cfg.hidden_dims), "activations": "elu"}]
    return {
        "seed": cfg.seed,
        "models": {
            "separate": True,
            "policy": {"class": "GaussianMixin", "clip_actions": True, "clip_log_std": True,
                       "min_log_std": -20, "max_log_std": 2, "initial_log_std": -0.69,
                       "network": network, "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "clip_actions": False, "network": network, "output": "ONE"},
        },
        "memory": {"class": "RandomMemory", "memory_size": cfg.rollout_steps},
        "agent": {
            "class": "PPO", "rollouts": cfg.rollout_steps, "learning_epochs": cfg.epochs,
            "mini_batches": cfg.mini_batches, "discount_factor": cfg.gamma, "lambda": cfg.gae_lambda,
            "learning_rate": cfg.learning_rate, "grad_norm_clip": cfg.max_grad_norm,
            "ratio_clip": cfg.clip, "value_clip": cfg.clip, "clip_predicted_values": True,
            "entropy_loss_scale": cfg.entropy_coef, "value_loss_scale": 1., "time_limit_bootstrap": True,
            "state_preprocessor": "RunningStandardScaler" if cfg.normalize_observations else None,
            "state_preprocessor_kwargs": {},
            "experiment": {"directory": str(output), "experiment_name": "metrics",
                           "write_interval": cfg.rollout_steps, "checkpoint_interval": 0},
        },
        "trainer": {"class": "SequentialTrainer", "timesteps": cfg.iterations * cfg.rollout_steps,
                    "headless": True, "close_environment_at_exit": False, "environment_info": "log"},
    }


class Backend:
    def train(self, env, cfg: TrainCfg, output: Path, resume: Path | None = None) -> Path:
        cfg.batch_size(env.unwrapped.num_envs)
        set_seed(cfg.seed)
        config = configuration(cfg, output)
        write_backend_config(output, config)
        runner = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), config)
        if resume:
            runner.agent.load(str(resume / CheckpointMeta.read(resume).checkpoint))
        runner.run()
        runner.agent.save(str(output / "model.pt"))
        return output / "model.pt"

    def load(self, env, folder: Path):
        meta = CheckpointMeta.read(folder)
        config = json.loads((folder / "backend.json").read_text())
        config["agent"]["experiment"].update(write_interval=0, checkpoint_interval=0)
        runner = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), config)
        runner.agent.load(str(folder / meta.checkpoint))
        runner.agent.set_running_mode("eval")

        def policy(observation):
            _, _, outputs = runner.agent.act(observation["policy"], timestep=0, timesteps=1)
            return outputs["mean_actions"]

        return policy
