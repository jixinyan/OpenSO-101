# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
import math
from pathlib import Path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from openso101.rl.config import CheckpointMeta, TrainCfg, write_backend_config
from openso101.rl.initialization import record_initial_std


def configuration(cfg: TrainCfg, output: Path, env):
    return {"params": {
        "seed": cfg.seed, "algo": {"name": "a2c_continuous"}, "model": {"name": "continuous_a2c_logstd"},
        "network": {
            "name": "actor_critic", "separate": False,
            "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None",
                                     "mu_init": {"name": "default"},
                                     "sigma_init": {"name": "const_initializer", "val": math.log(cfg.initial_noise_std)}, "fixed_sigma": True}},
            "mlp": {"units": list(cfg.hidden_dims), "activation": "elu", "d2rl": False,
                    "initializer": {"name": "default"}, "regularizer": {"name": None}},
        },
        "config": {
            "name": "openso101", "env_name": "openso101_rlgpu", "device": env.unwrapped.device,
            "device_name": env.unwrapped.device, "num_actors": env.unwrapped.num_envs,
            "ppo": True, "mixed_precision": False, "normalize_input": cfg.normalize_observations,
            "normalize_value": True, "normalize_advantage": True, "reward_shaper": {"scale_value": 1.},
            "gamma": cfg.gamma, "tau": cfg.gae_lambda, "learning_rate": cfg.learning_rate,
            "lr_schedule": "constant", "max_epochs": cfg.iterations, "save_best_after": 0,
            "save_frequency": cfg.iterations, "grad_norm": cfg.max_grad_norm,
            "entropy_coef": cfg.entropy_coef, "truncate_grads": True, "e_clip": cfg.clip,
            "horizon_length": cfg.rollout_steps, "minibatch_size": cfg.batch_size(env.unwrapped.num_envs),
            "mini_epochs": cfg.epochs, "critic_coef": 1., "clip_value": True, "bounds_loss_coef": 0.0001,
            "train_dir": str(output), "full_experiment_name": "metrics",
            "player": {"deterministic": True},
        },
    }}


def create_runner(env, config):
    wrapped = RlGamesVecEnvWrapper(env, env.unwrapped.device, 10., 1.)
    vecenv.register("OpenSO101GpuEnv", lambda name, count, **kwargs: RlGamesGpuEnv(name, count, **kwargs))
    env_configurations.register("openso101_rlgpu", {"vecenv_type": "OpenSO101GpuEnv", "env_creator": lambda **kwargs: wrapped})
    runner = Runner(IsaacAlgoObserver())
    runner.load(config)
    runner.reset()
    return runner


class Backend:
    def train(self, env, cfg: TrainCfg, output: Path, resume: Path | None = None) -> Path:
        config = configuration(cfg, output, env)
        write_backend_config(output, config)
        runner = create_runner(env, config)
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        if resume:
            agent.restore(str(resume / CheckpointMeta.read(resume).checkpoint))
            agent.max_epochs = agent.epoch_num + cfg.iterations
        record_initial_std(output, cfg, agent.model.a2c_network.sigma.exp(), resumed=resume is not None)
        agent.train()
        agent.save(str(output / "model"))
        return output / "model.pth"

    def load(self, env, folder: Path):
        meta = CheckpointMeta.read(folder)
        config = json.loads((folder / "backend.json").read_text())
        config["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = create_runner(env, config)
        player = runner.create_player()
        player.restore(str(folder / meta.checkpoint))
        player.reset()
        player.has_batch_dimension = True
        return lambda observation: player.get_action(observation["policy"], is_deterministic=True)
