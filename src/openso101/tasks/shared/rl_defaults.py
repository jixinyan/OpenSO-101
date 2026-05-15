# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO101-specific RL defaults for manipulation tasks.

Reward design follows the upstream MuammerBay/isaac_so_arm101 pattern:
minimal AND-conjunctions, dense chain rewards (reach → grip-near → lift →
goal-track), height-only gating once the cube is airborne. Values are
adapted for SO101's ~35cm starting EE-cube distance (vs upstream's ~10cm)
via a coarse+fine two-tanh reach kernel.

Smoothness penalties start at 0 and curriculum in once lift fires; otherwise
they suppress exploration before the chain bootstraps."""

# Reach reward: coarse kernel for long-range gradient, fine for grasp approach
SO101_REACH_REWARD_COARSE_STD = 0.20
SO101_REACH_REWARD_COARSE_WEIGHT = 0.5
SO101_REACH_REWARD_STD = 0.05
SO101_CLOSE_GRIPPER_NEAR_THRESHOLD = 0.08
SO101_CLOSE_GRIPPER_CLOSED_STD = 0.12

SO101_CONTROLLED_OBJECT_MIN_HEIGHT = 0.04
SO101_CONTROLLED_GRASP_DISTANCE_THRESHOLD = 0.08
SO101_GRIPPER_CLOSED_THRESHOLD = 0.12

SO101_GOAL_TRACKING_STD = 0.30
SO101_GOAL_TRACKING_FINE_STD = 0.05

# Smoothness penalties: disabled at step 0; curriculum ramps in after lift fires
SO101_ACTION_RATE_WEIGHT = 0.0
SO101_JOINT_VEL_WEIGHT = 0.0
SO101_JOINT_POS_DELTA_WEIGHT = 0.0
SO101_SMOOTHNESS_CURRICULUM_WEIGHT = -1.0e-4
SO101_SMOOTHNESS_CURRICULUM_STEPS = 48_000

SO101_PPO_NUM_STEPS_PER_ENV = 96
SO101_PPO_INIT_NOISE_STD = 1.0
SO101_PPO_NOISE_STD_TYPE: str = "log"
"""Use log-parameterization for the PPO action-noise std so sigma = exp(param)
stays strictly positive. The default "scalar" lets sigma drift negative under
gradient noise and crashes torch.distributions.Normal.sample(). Observed crash:
2026-05-14 lift training, iter 79, RuntimeError "normal expects all elements
of std >= 0.0"."""
SO101_PPO_ENTROPY_COEF = 0.005
SO101_PPO_GAMMA = 0.98


__all__ = [
    "SO101_REACH_REWARD_COARSE_STD",
    "SO101_REACH_REWARD_COARSE_WEIGHT",
    "SO101_REACH_REWARD_STD",
    "SO101_CLOSE_GRIPPER_NEAR_THRESHOLD",
    "SO101_CLOSE_GRIPPER_CLOSED_STD",
    "SO101_CONTROLLED_OBJECT_MIN_HEIGHT",
    "SO101_CONTROLLED_GRASP_DISTANCE_THRESHOLD",
    "SO101_GRIPPER_CLOSED_THRESHOLD",
    "SO101_GOAL_TRACKING_STD",
    "SO101_GOAL_TRACKING_FINE_STD",
    "SO101_ACTION_RATE_WEIGHT",
    "SO101_JOINT_VEL_WEIGHT",
    "SO101_JOINT_POS_DELTA_WEIGHT",
    "SO101_SMOOTHNESS_CURRICULUM_WEIGHT",
    "SO101_SMOOTHNESS_CURRICULUM_STEPS",
    "SO101_PPO_NUM_STEPS_PER_ENV",
    "SO101_PPO_INIT_NOISE_STD",
    "SO101_PPO_NOISE_STD_TYPE",
    "SO101_PPO_ENTROPY_COEF",
    "SO101_PPO_GAMMA",
]
