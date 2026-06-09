# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Reward functions for pick-and-lift (sentinel-style delta shaping).

This task follows the sentinel ``PickAndLiftReward`` design: a single fixed
goal in the air and *delta-distance* shaping that rewards progress rather
than position. Concretely the per-step reward is::

    pregrasp_approach   (active while NOT grasping):  weight * Delta(eef -> obj)
    grasp_hold          (active while grasping):      grasped_reward (1.0)
    carry_to_goal       (active while grasping):      weight * Delta(obj -> goal)
    success_bonus       (terminal):                   reached goal AND grasped

Delta shaping closes the "hover near target and farm reward" exploit that
absolute tanh-distance shaping permits: holding position yields zero, only
*reducing* the distance pays. The grasp-mode gating (pregrasp XOR carry) and
the fresh-reset guard come from the pure, unit-tested core in
``openso101.shaping``; the functions here only gather the tensors.

The previous 3-stage curriculum reward chain (lift/carry/place with
height-only gating) was removed in favour of this single-goal design; the
goal location is frozen by the command term (``lock_stage``), not by a
per-step reward gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from openso101.shaping import (
    delta_progress_reward,
    resolve_prev,
    shaping_valid_mask,
)
from openso101.tasks.shared.grasp import object_grasped_by_jaws

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _eef_to_object_distance(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame distance from the end-effector frame to the object, per env."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    return torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)


def _object_to_goal_distance(
    env: "ManagerBasedRLEnv",
    command_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World-frame distance from the object to the commanded goal, per env.

    The command exposes the goal in the robot root frame; we transform it to
    world before differencing against the object's world position (same
    convention as :func:`shared.rewards.object_reached_goal_in_air`).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    goal_b = env.command_manager.get_command(command_name)[:, :3]
    goal_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], goal_b
    )
    return torch.norm(goal_w - obj.data.root_pos_w[:, :3], dim=1)


def _shaped_delta(
    env: "ManagerBasedRLEnv",
    state_key: str,
    cur_dist: torch.Tensor,
    mode_active: torch.Tensor,
) -> torch.Tensor:
    """Per-env delta-progress reward with grasp-mode + reset gating.

    Maintains the previous distance and previous mode flag in ``env.extras``
    (the same persistent-state pattern as ``shared.rewards.joint_pos_delta_l2``)
    under ``state_key``-prefixed entries so multiple shaping terms never stomp
    each other. Returns the *unweighted* delta; the ``RewardTerm`` weight is
    the shaping coefficient.
    """
    dist_key = f"{state_key}_dist"
    mode_key = f"{state_key}_mode"

    prev_dist = resolve_prev(env.extras.get(dist_key), cur_dist)
    prev_mode = env.extras.get(mode_key)
    if prev_mode is None or prev_mode.shape != mode_active.shape:
        prev_mode = mode_active.clone()

    # episode_length_buf is incremented before reward computation, so a value
    # of <=1 marks the first reward step of a fresh episode, where prev_dist is
    # stale from the previous episode and must not be differenced.
    fresh_reset = env.episode_length_buf <= 1
    valid = shaping_valid_mask(mode_active, prev_mode, fresh_reset)
    reward = delta_progress_reward(prev_dist, cur_dist, coeff=1.0, valid_mask=valid)

    env.extras[dist_key] = cur_dist.detach().clone()
    env.extras[mode_key] = mode_active.detach().clone()
    return reward


def pregrasp_approach_shaping(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Delta-shaping on (eef -> object) distance, active only while NOT grasping.

    Drives the gripper toward the cube before contact. Once a contact-confirmed
    grasp is achieved the term goes silent (mode flips to carry); the grasp
    onset step yields zero (mode-transition guard) so there's no spurious
    reward from the eef "snapping" onto the cube.
    """
    grasping = object_grasped_by_jaws(env, force_threshold)
    cur = _eef_to_object_distance(env, object_cfg, ee_frame_cfg)
    return _shaped_delta(env, "_pp_pregrasp", cur, ~grasping)


def carry_to_goal_shaping(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    force_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Delta-shaping on (object -> goal) distance, active only while grasping.

    Because the goal is fixed in the air, reducing this distance *requires*
    lifting and carrying the held cube — there is no on-table drag shortcut.
    Silent while not grasping; the grasp-release step yields zero.
    """
    grasping = object_grasped_by_jaws(env, force_threshold)
    cur = _object_to_goal_distance(env, command_name, robot_cfg, object_cfg)
    return _shaped_delta(env, "_pp_carry", cur, grasping)


def grasp_onset_bonus(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """One-shot reward (1.0) on the step a grasp transitions False->True, per env.

    Returns 1.0 exactly on the first step the contact-confirmed grasp predicate
    flips from not-grasping to grasping, and 0.0 otherwise (including every
    subsequent step the grasp is held — that steady-state reward is the job of
    ``grasped_reward``/``grasp_hold``). This injects a discrete reward at the
    reach->grasp boundary so the policy gets a sharp, unambiguous credit for
    *achieving* the grasp rather than only for maintaining it.

    Previous grasp state is tracked in ``env.extras`` under a private key,
    mirroring the ``_shaped_delta`` persistent-state pattern: ``resolve_prev``
    seeds it on first call, and a fresh-reset guard (``episode_length_buf <= 1``)
    yields 0 so a grasp held across an episode boundary doesn't fire a spurious
    onset on the first step of the new episode.
    """
    grasping = object_grasped_by_jaws(env, force_threshold)
    prev = resolve_prev(env.extras.get("_pp_grasp_prev"), grasping.float())
    prev_grasping = prev > 0.5

    # Onset = not-grasping last step AND grasping this step. Suppressed on a
    # fresh episode reset where prev is stale from the previous episode.
    fresh_reset = env.episode_length_buf <= 1
    onset = grasping & (~prev_grasping) & (~fresh_reset)

    env.extras["_pp_grasp_prev"] = grasping.float().detach().clone()
    return onset.float()


__all__ = [
    "pregrasp_approach_shaping",
    "carry_to_goal_shaping",
    "grasp_onset_bonus",
]
