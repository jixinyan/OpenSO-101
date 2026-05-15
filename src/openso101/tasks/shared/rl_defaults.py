# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""SO-101 RL reward design defaults shared by all built-in tasks.

SKELETON: real values will be ported from
`/data/safe_sim2real/src/safe_sim2real/tasks/so101_rl_defaults.py` once
the source is finalized. The legacy module documents the upstream
MuammerBay/isaac_so_arm101 chain-reward design (coarse + fine reach,
grip-near, lift, goal-track) tuned for SO-101 workspace scale.
"""

from __future__ import annotations

# Reach reward: coarse kernel for long-range gradient, fine for grasp approach
SO101_REACH_REWARD_COARSE_STD: float = 0.0
SO101_REACH_REWARD_COARSE_WEIGHT: float = 0.0
SO101_REACH_REWARD_STD: float = 0.0

# Grip-near gating
SO101_CLOSE_GRIPPER_NEAR_THRESHOLD: float = 0.0
SO101_CLOSE_GRIPPER_CLOSED_STD: float = 0.0

# Controlled-grasp predicates
SO101_CONTROLLED_OBJECT_MIN_HEIGHT: float = 0.0
SO101_CONTROLLED_GRASP_DISTANCE_THRESHOLD: float = 0.0
SO101_GRIPPER_CLOSED_THRESHOLD: float = 0.0

# Goal tracking
SO101_GOAL_TRACKING_STD: float = 0.0
SO101_GOAL_TRACKING_FINE_STD: float = 0.0

# Smoothness penalties: disabled at step 0; curriculum ramps in after lift fires
SO101_ACTION_RATE_WEIGHT: float = 0.0
SO101_JOINT_VEL_WEIGHT: float = 0.0
SO101_JOINT_POS_DELTA_WEIGHT: float = 0.0
SO101_SMOOTHNESS_CURRICULUM_WEIGHT: float = 0.0
SO101_SMOOTHNESS_CURRICULUM_STEPS: int = 0

# PPO training defaults
SO101_PPO_NUM_STEPS_PER_ENV: int = 0
SO101_PPO_INIT_NOISE_STD: float = 0.0
SO101_PPO_NOISE_STD_TYPE: str = "log"
SO101_PPO_ENTROPY_COEF: float = 0.0
SO101_PPO_GAMMA: float = 0.0


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
