# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from openso101.robots.so101.ik import differential_ik
from openso101.scenes.models import Entity, Goal, Pose, SceneSpec, Task
from openso101.scenes.task import SuccessTracker, evaluate_goals
from openso101.sim2real.domain_randomization.action import (
    ActionDelayBuffer,
    FirstOrderActionLag,
)


def test_ik_reduces_linearized_error_and_respects_limits():
    jacobian = torch.tensor([[[0.2, 0, 0, 0, 0], [0, 0.2, 0, 0, 0], [0, 0, 0.2, 0, 0], [0, 0, 0, 1, 0]]])
    q = torch.zeros(1, 5)
    limits = torch.tensor([[[-1., 1.]] * 5])
    delta = torch.tensor([[0.005, -0.005, 0.005, 0.01]])
    result = differential_ik(jacobian, delta, q, limits, dt=0.1)
    error = delta - (jacobian @ result.unsqueeze(-1)).squeeze(-1)
    assert torch.linalg.vector_norm(error) < torch.linalg.vector_norm(delta) * 0.1
    assert (result.abs() <= 0.1).all()
    with pytest.raises(ValueError, match="有限"):
        differential_ik(jacobian * float("nan"), delta, q, limits, dt=0.1)


def test_delay_partial_reset_preserves_other_history():
    delay = ActionDelayBuffer(2, 1, 2, min_delay=2)
    delay.reset()
    delay.step(torch.tensor([[1.], [10.]]))
    delay.step(torch.tensor([[2.], [20.]]))
    delay.reset([0])
    assert torch.equal(delay.step(torch.tensor([[3.], [30.]])), torch.tensor([[3.], [10.]]))
    assert torch.equal(delay.step(torch.tensor([[4.], [40.]])), torch.tensor([[3.], [20.]]))


def test_lag_partial_reset_preserves_other_history():
    lag = FirstOrderActionLag(2, 1, alpha_min=0.5, alpha_max=0.5)
    lag.reset()
    lag.step(torch.tensor([[1.], [10.]]))
    lag.reset([0])
    assert torch.equal(lag.step(torch.tensor([[3.], [20.]])), torch.tensor([[3.], [15.]]))


def task_scene(predicate):
    obj = Entity(entity_id="apple", asset_uid="a" * 32, asset_sha256="a" * 64,
                 dimensions_m=(0.04, 0.04, 0.04), pose=Pose(position=(0.2, 0, 0.02)))
    target = Entity(entity_id="tray", asset_uid="b" * 32, asset_sha256="b" * 64,
                    dimensions_m=(0.12, 0.12, 0.04), pose=Pose(position=(0.3, 0, 0.02)), dynamic=False)
    goal = Goal(object_id="apple", predicate=predicate, target_id="tray",
                region_bounds_m=((-0.05, -0.05, 0), (0.05, 0.05, 0.1)) if predicate == "inside" else None)
    return SceneSpec(scene_id="placement", entities=(obj, target), task=Task(
        task_id="place", object_id="apple", goal_position_m=(0.3, 0, 0.06), goals=(goal,), settle_seconds=0.5,
    ))


@pytest.mark.parametrize("predicate", ["inside", "on_top"])
def test_relation_requires_full_object_release_and_stability(predicate):
    spec = task_scene(predicate)
    states = {"apple": np.array([0.3, 0, 0.06, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              "tray": np.array([0.3, 0, 0.02, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    tracker = SuccessTracker(spec)
    assert not tracker.update(states, False, 0.5)["success"]
    assert not tracker.update(states, True, 0.25)["success"]
    assert tracker.update(states, True, 0.25)["success"]
    states["apple"][10] = 0.2
    assert not tracker.update(states, True, 0.5)["success"]
    states["apple"][10] = 0
    states["apple"][0] = 0.35
    assert not evaluate_goals(spec, states, True)["instant_success"]
