# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.spatial.transform import Rotation

from .models import Goal, SceneSpec


def evaluate_goals(spec: SceneSpec, states: dict[str, np.ndarray], gripper_open: bool) -> dict:
    entities = {entity.entity_id: entity for entity in spec.entities}
    for entity_id in entities:
        if entity_id not in states:
            raise KeyError(f"missing state for entity: {entity_id}")
        state = np.asarray(states[entity_id])
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError(f"实体状态必须包含 13 个有限数值：{entity_id}")
        if not np.isclose(np.dot(state[3:7], state[3:7]), 1.0, atol=1e-4):
            raise ValueError(f"实体 quaternion 必须为单位 quaternion：{entity_id}")
    goals = spec.task.goals or (Goal(object_id=spec.task.object_id, position_m=spec.task.goal_position_m),)
    conditions = []
    for goal in goals:
        state = np.asarray(states[goal.object_id])
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError("实体状态必须包含 position、quaternion、linear velocity 和 angular velocity")
        position = state[:3]
        half = np.asarray(entities[goal.object_id].dimensions_m) / 2
        rotation = Rotation.from_quat(state[3:7], scalar_first=True).as_matrix()
        if goal.predicate == "at":
            reached = np.linalg.norm(position - goal.position_m) <= spec.task.position_tolerance_m
        else:
            target = np.asarray(states[goal.target_id])
            target_rotation = Rotation.from_quat(target[3:7], scalar_first=True).as_matrix()
            local_position = target_rotation.T @ (position - target[:3])
            half_local = np.abs(target_rotation.T @ rotation) @ half
            if goal.predicate == "inside":
                lower, upper = np.asarray(goal.region_bounds_m)
                reached = np.all(local_position - half_local >= lower) and np.all(local_position + half_local <= upper)
            else:
                target_half = np.asarray(entities[goal.target_id].dimensions_m) / 2
                reached = (
                    np.all(np.abs(local_position[:2]) + half_local[:2] <= target_half[:2])
                    and abs(local_position[2] - half_local[2] - target_half[2]) <= spec.task.position_tolerance_m
                )
        stable = (
            np.linalg.norm(state[7:10]) <= spec.task.max_linear_speed_m_s
            and np.linalg.norm(state[10:13]) <= spec.task.max_angular_speed_rad_s
        )
        conditions.append({"object_id": goal.object_id, "reached": bool(reached), "stable": bool(stable)})
    return {"conditions": conditions, "instant_success": bool(
        all(item["reached"] and item["stable"] for item in conditions)
        and (gripper_open or not spec.task.require_released)
    )}


class SuccessTracker:
    def __init__(self, spec: SceneSpec):
        self.spec = spec
        self.elapsed = 0.0

    def reset(self):
        self.elapsed = 0.0

    def update(self, states, gripper_open, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须为正数有限数值")
        report = evaluate_goals(self.spec, states, gripper_open)
        self.elapsed = self.elapsed + dt if report["instant_success"] else 0.0
        report["held_seconds"] = self.elapsed
        report["success"] = self.elapsed + 1e-9 >= self.spec.task.settle_seconds
        return report
