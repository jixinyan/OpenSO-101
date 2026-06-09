# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Unit tests for the pure delta-distance shaping primitives.

These exercise the sentinel-style reward core (``openso101.shaping``)
with plain tensors — no SimulationApp, so they run in plain pytest.
"""

from __future__ import annotations

import torch

from openso101.shaping import (
    delta_progress_reward,
    resolve_prev,
    shaping_valid_mask,
)


def test_delta_positive_when_closer_and_valid():
    prev = torch.tensor([1.0, 0.5])
    cur = torch.tensor([0.8, 0.5])  # env 0 got 0.2 closer; env 1 unchanged
    valid = torch.tensor([True, True])
    r = delta_progress_reward(prev, cur, coeff=2.0, valid_mask=valid)
    assert torch.allclose(r, torch.tensor([0.4, 0.0]))


def test_delta_negative_when_moving_away():
    # Moving away from the target is penalised (sentinel allows negative shaping).
    prev = torch.tensor([0.5])
    cur = torch.tensor([0.9])
    valid = torch.tensor([True])
    r = delta_progress_reward(prev, cur, coeff=1.0, valid_mask=valid)
    assert torch.allclose(r, torch.tensor([-0.4]))


def test_delta_zeroed_where_invalid():
    prev = torch.tensor([1.0, 1.0])
    cur = torch.tensor([0.5, 0.5])
    valid = torch.tensor([True, False])
    r = delta_progress_reward(prev, cur, coeff=1.0, valid_mask=valid)
    assert torch.allclose(r, torch.tensor([0.5, 0.0]))


def test_valid_mask_requires_same_mode_both_steps():
    mode_now = torch.tensor([True, True, False, False])
    mode_prev = torch.tensor([True, False, True, False])
    fresh = torch.tensor([False, False, False, False])
    mask = shaping_valid_mask(mode_now, mode_prev, fresh)
    # Only valid where the mode was active on BOTH steps.
    assert mask.tolist() == [True, False, False, False]


def test_valid_mask_false_on_fresh_reset():
    mode_now = torch.tensor([True, True])
    mode_prev = torch.tensor([True, True])
    fresh = torch.tensor([True, False])
    mask = shaping_valid_mask(mode_now, mode_prev, fresh)
    assert mask.tolist() == [False, True]


def test_mode_transition_yields_zero_reward():
    # Simulate the grasp-onset step: mode just became active, so prev_mode is
    # False -> no spurious delta even though the distance changed a lot.
    prev_dist = torch.tensor([0.9])
    cur_dist = torch.tensor([0.1])
    mode_now = torch.tensor([True])
    mode_prev = torch.tensor([False])
    fresh = torch.tensor([False])
    mask = shaping_valid_mask(mode_now, mode_prev, fresh)
    r = delta_progress_reward(prev_dist, cur_dist, coeff=2.0, valid_mask=mask)
    assert torch.allclose(r, torch.tensor([0.0]))


def test_resolve_prev_seeds_to_current_when_absent():
    cur = torch.tensor([1.0, 2.0])
    prev = resolve_prev(None, cur)
    assert torch.allclose(prev, cur)
    # First delta is therefore exactly zero.
    r = delta_progress_reward(prev, cur, coeff=5.0, valid_mask=torch.tensor([True, True]))
    assert torch.allclose(r, torch.zeros(2))


def test_resolve_prev_reseeds_on_shape_change():
    cur = torch.tensor([1.0, 2.0, 3.0])
    stale = torch.tensor([0.0])  # wrong shape (e.g. num_envs changed)
    prev = resolve_prev(stale, cur)
    assert prev.shape == cur.shape
    assert torch.allclose(prev, cur)


def test_resolve_prev_keeps_valid_prev():
    cur = torch.tensor([1.0, 2.0])
    prev_in = torch.tensor([0.5, 1.5])
    prev = resolve_prev(prev_in, cur)
    assert torch.allclose(prev, prev_in)
