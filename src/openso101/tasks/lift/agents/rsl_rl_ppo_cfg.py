# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PPO RSL-RL runner cfg for the Lift task."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from openso101.tasks.shared.rl_defaults import (
    SO101_PPO_ENTROPY_COEF,
    SO101_PPO_GAMMA,
    SO101_PPO_INIT_NOISE_STD,
    SO101_PPO_NOISE_STD_TYPE,
    SO101_PPO_NUM_STEPS_PER_ENV,
)


@configclass
class LiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = SO101_PPO_NUM_STEPS_PER_ENV
    max_iterations = 1500
    save_interval = 50
    experiment_name = "lift"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=SO101_PPO_INIT_NOISE_STD,
        noise_std_type=SO101_PPO_NOISE_STD_TYPE,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=SO101_PPO_ENTROPY_COEF,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=SO101_PPO_GAMMA,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
