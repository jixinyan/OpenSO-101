# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""PPO RSL-RL runner cfg for the PickPlace task."""

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
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Long rollouts (96 steps) so individual envs can capture grasp -> lift -> carry
    # transitions inside one rollout instead of bootstrapping mid-skill. Episode length
    # is 400 steps; ~4 rollouts per episode keeps the value bootstrap reasonable.
    num_steps_per_env = SO101_PPO_NUM_STEPS_PER_ENV
    max_iterations = 2000
    save_interval = 100
    experiment_name = "pick_place"
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
        # Mild exploration boost over the default 0.001--0.002. Sparse-reward manipulation
        # needs entropy retained long enough for grasp/release sub-skills to emerge.
        entropy_coef=SO101_PPO_ENTROPY_COEF,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        # With gamma=0.99 (SO101_PPO_GAMMA) the effective horizon is
        # 1/(1-gamma)=~100 steps, enough for the grasp -> carry -> goal sparse
        # rewards to propagate back across a carry without early dense guidance
        # dominating the critic. (Raised from 0.98 / ~50-step horizon.)
        gamma=SO101_PPO_GAMMA,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
