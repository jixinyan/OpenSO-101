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
# discounts with gamma = 0.99 (SO101_PPO_GAMMA).
#
# AUDIT HYPOTHESIS (needs a training run to validate): the pregrasp/carry/
# grasp-hold/onset magnitudes below are NOT analytically derived — they come
# from a reward audit that judged the previous values (pregrasp=1, carry=2,
# hold=1) too weak relative to the terminal goal bonus to reliably pull the
# policy through the grasp transition. The previous "sentinel-balance"
# derivation assumed gamma=0.98 and made the discounted infinite grasp-hold
# sum (hold_w * dt / (1 - gamma)) exactly equal the terminal bonus
# (goal_w * dt). With gamma=0.99 and the reweighted terms that exact equality
# no longer holds and is NOT re-derived here: the balance is now tuned
# EMPIRICALLY and these numbers are starting hypotheses, not a closed-form
# optimum. The actual delivery driver remains the carry delta-shaping
# (positive every step the held cube nears the goal), now boosted so the
# pull toward the goal out-values lingering in the grasp-hold reward; the
# one-shot grasp-onset bonus injects a discrete reward at the
# False->True grasp transition to sharpen the reach->grasp boundary.
SO101_PICK_PREGRASP_COEFF = 30.0  # Delta(eef -> obj) gain, active while not grasping (audit hypothesis; needs training run)
SO101_PICK_CARRY_COEFF = 12.0  # Delta(obj -> goal) gain, active while grasping (audit hypothesis; needs training run)
SO101_PICK_GRASP_HOLD_WEIGHT = 0.2  # per-step contact-confirmed-grasp reward (audit hypothesis; needs training run)
SO101_PICK_GOAL_BONUS = 50.0  # terminal: reached goal sphere AND still grasped
SO101_PICK_GRASP_ONSET_BONUS = 5.0  # one-shot reward on the first step a grasp is confirmed (audit hypothesis; needs training run)

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
# gamma 0.98 -> 0.99: a safe horizon fix. 1/(1-0.99) = 100-step effective
# horizon, enough for the ~400-step pick-and-lift episode's terminal goal
# bonus to propagate back through a carry without dense early shaping
# dominating the critic. (0.98 gave only a ~50-step horizon.)
SO101_PPO_GAMMA = 0.99

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
    "SO101_PICK_GRASP_ONSET_BONUS",
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
