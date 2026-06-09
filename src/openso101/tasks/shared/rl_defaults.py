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

# Pick-and-lift (sentinel PickAndLiftReward) delta-shaping weights.
#
# These are used directly as RewardTerm weights. The RewardManager multiplies
# every term by `weight * dt` (dt = decimation * sim.dt = 0.02 s), and PPO
# discounts with gamma = 0.98. That uniform scaling preserves the sentinel
# balance: the discounted value of holding the cube forever
# (grasp_w * dt / (1 - gamma) = 1 * 0.02 / 0.02 = 1.0) equals the terminal goal
# bonus (goal_w * dt = 50 * 0.02 = 1.0). The actual delivery driver is the
# carry delta-shaping (positive every step the held cube nears the goal), not
# the terminal bonus.
SO101_PICK_PREGRASP_COEFF = 1.0  # Delta(eef -> obj) gain, active while not grasping
SO101_PICK_CARRY_COEFF = 2.0  # Delta(obj -> goal) gain, active while grasping
SO101_PICK_GRASP_HOLD_WEIGHT = 1.0  # per-step contact-confirmed-grasp reward
SO101_PICK_GOAL_BONUS = 50.0  # terminal: reached goal sphere AND still grasped

# Smoothness penalties.
#
# joint_vel is a "physical sanity" penalty: ACTIVE from step 0 (not
# curriculum-ramped). Without it, the arm jitters chaotically and swats
# the cube off the table before the policy ever learns to reach it.
#
# action_rate and joint_pos_delta are pure exploration-suppressors: they
# stay on the curriculum ramp so they don't kill exploration before lift fires.
SO101_ACTION_RATE_WEIGHT = 0.0
SO101_JOINT_VEL_WEIGHT = -1.0e-4
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

# ---------------------------------------------------------------------------
# Distillation defaults (rsl_rl.algorithms.Distillation)
# ---------------------------------------------------------------------------
# Same hidden-dim shape as PPO so a freshly-trained PPO checkpoint can be
# loaded as the teacher without architecture mismatch. Student is a copy of
# the teacher architecture by default; users who want a smaller, faster
# student can override `student_hidden_dims` at the task-config level.
SO101_DISTILL_HIDDEN_DIMS = [256, 128, 64]
SO101_DISTILL_ACTIVATION = "elu"
SO101_DISTILL_INIT_NOISE_STD = SO101_PPO_INIT_NOISE_STD
SO101_DISTILL_NOISE_STD_TYPE: str = SO101_PPO_NOISE_STD_TYPE
SO101_DISTILL_NUM_STEPS_PER_ENV = SO101_PPO_NUM_STEPS_PER_ENV
SO101_DISTILL_LEARNING_RATE = 1.0e-4
SO101_DISTILL_NUM_LEARNING_EPOCHS = 5
SO101_DISTILL_GRADIENT_LENGTH = 32
SO101_DISTILL_MAX_GRAD_NORM = 1.0
SO101_DISTILL_LOSS_TYPE: str = "mse"


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
    "SO101_PICK_PREGRASP_COEFF",
    "SO101_PICK_CARRY_COEFF",
    "SO101_PICK_GRASP_HOLD_WEIGHT",
    "SO101_PICK_GOAL_BONUS",
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
    "SO101_DISTILL_HIDDEN_DIMS",
    "SO101_DISTILL_ACTIVATION",
    "SO101_DISTILL_INIT_NOISE_STD",
    "SO101_DISTILL_NOISE_STD_TYPE",
    "SO101_DISTILL_NUM_STEPS_PER_ENV",
    "SO101_DISTILL_LEARNING_RATE",
    "SO101_DISTILL_NUM_LEARNING_EPOCHS",
    "SO101_DISTILL_GRADIENT_LENGTH",
    "SO101_DISTILL_MAX_GRAD_NORM",
    "SO101_DISTILL_LOSS_TYPE",
]
