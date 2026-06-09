# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Runtime smoke test for the sentinel pick-and-lift reward stack.

Builds a tiny (4-env) PickPlace env and steps it with random actions to
confirm the delta-shaping reward terms execute end to end: grasp-sensor reads,
goal-frame transform, and the ``env.extras`` previous-distance / mode state
machine. This catches wiring/runtime errors the pure unit tests (which use
plain tensors) cannot. Needs a GPU + SimulationApp, so it is skipped when
isaaclab is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab_tasks")

import gymnasium as gym
import torch

import openso101.tasks.pick_place  # noqa: F401 — registers OpenSO101-PickPlace-v0
from openso101.tasks.pick_place import PickPlaceEnvCfg


def _tiny_cfg() -> PickPlaceEnvCfg:
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = 4
    # Shrink PhysX GPU buffers sized for 4096-env training so 4 envs fit on a
    # small GPU (same revert teleop does for num_envs=1).
    cfg.sim.physx.gpu_collision_stack_size = 64 * 1024 * 1024
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 1024
    return cfg


def test_pick_lift_rewards_step_without_error():
    env = gym.make("OpenSO101-PickPlace-v0", cfg=_tiny_cfg())
    try:
        env.reset()
        unwrapped = env.unwrapped
        action = torch.zeros(
            (unwrapped.num_envs, unwrapped.action_manager.total_action_dim),
            device=unwrapped.device,
        )
        for _ in range(20):
            _obs, rew, _term, _trunc, _info = env.step(action)
            assert torch.isfinite(rew).all(), "non-finite reward emitted"
            assert rew.shape[0] == 4
    finally:
        env.close()

    # The delta-shaping terms stash per-env previous-distance/mode state in
    # env.extras; their presence confirms both shaping terms actually ran.
    extras = env.unwrapped.extras
    for key in ("_pp_pregrasp_dist", "_pp_pregrasp_mode", "_pp_carry_dist", "_pp_carry_mode"):
        assert key in extras, f"missing shaping state {key!r} — reward term did not run"
        assert extras[key].shape[0] == 4
