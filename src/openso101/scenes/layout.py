# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .bundle import entity_bounds, validate_layout
from .catalog import AssetCatalog
from .models import Pose, SceneSpec


def solve_layout(spec: SceneSpec, catalog: AssetCatalog, *, locked=(), clearance_m=0.01) -> SceneSpec:
    if clearance_m < 0 or not np.isfinite(clearance_m):
        raise ValueError("clearance_m 必须为非负有限数值")
    ids = [entity.entity_id for entity in spec.entities]
    if set(locked) - set(ids):
        raise ValueError("locked 包含未知实体")
    n = len(ids)
    shapes = [entity_bounds(entity) - np.asarray(entity.pose.position) for entity in spec.entities]
    reset_shapes = [shape + np.asarray(entity.reset_translation_m) for shape, entity in zip(shapes, spec.entities)]
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        zi = reset_shapes[i][:, 2] + spec.entities[i].pose.position[2]
        zj = reset_shapes[j][:, 2] + spec.entities[j].pose.position[2]
        if zi[0] < zj[1] and zj[0] < zi[1]:
            pairs.append((i, j))
    count = 4 * n + 4 * len(pairs)
    lower = np.zeros(count)
    upper = np.full(count, np.inf)
    integrality = np.zeros(count)
    objective = np.zeros(count)
    objective[2 * n:4 * n] = 1
    table_lower = np.array(spec.table.center_xy_m) - np.array(spec.table.size_xy_m) / 2
    table_upper = np.array(spec.table.center_xy_m) + np.array(spec.table.size_xy_m) / 2
    rows, row_lower, row_upper = [], [], []

    def constraint(coefficients, lo=-np.inf, hi=np.inf):
        row = np.zeros(count)
        for index, value in coefficients.items():
            row[index] = value
        rows.append(row)
        row_lower.append(lo)
        row_upper.append(hi)

    for i, entity in enumerate(spec.entities):
        for axis in range(2):
            index = 2 * i + axis
            lo = table_lower[axis] - reset_shapes[i][0, axis]
            hi = table_upper[axis] - reset_shapes[i][1, axis]
            if entity.entity_id in locked:
                if not lo <= entity.pose.position[axis] <= hi:
                    raise ValueError(f"指定位置超出桌面：{entity.entity_id}")
                lo = hi = entity.pose.position[axis]
            lower[index], upper[index] = lo, hi
            deviation = 2 * n + index
            constraint({index: 1, deviation: -1}, hi=entity.pose.position[axis])
            constraint({index: -1, deviation: -1}, hi=-entity.pose.position[axis])
    for pair_index, (i, j) in enumerate(pairs):
        binary_start = 4 * n + 4 * pair_index
        for direction, (first, second, axis) in enumerate(((i, j, 0), (j, i, 0), (i, j, 1), (j, i, 1))):
            binary = binary_start + direction
            integrality[binary] = 1
            upper[binary] = 1
            gap = reset_shapes[first][1, axis] - reset_shapes[second][0, axis] + clearance_m
            big_m = upper[2 * first + axis] - lower[2 * second + axis] + gap
            constraint({2 * first + axis: 1, 2 * second + axis: -1, binary: big_m}, hi=big_m - gap)
        constraint(dict.fromkeys(range(binary_start, binary_start + 4), 1), lo=1)
    if (lower > upper).any():
        raise ValueError("物体尺寸和 reset 范围无法满足桌面限制")
    result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(np.asarray(rows), row_lower, row_upper),
                  options={"time_limit": 30.})
    if not result.success:
        raise ValueError(f"布局求解失败：{result.message}")
    entities = []
    for i, entity in enumerate(spec.entities):
        position = (*result.x[2 * i:2 * i + 2], entity.pose.position[2])
        entities.append(entity.model_copy(update={"pose": Pose(position=position, quaternion_wxyz=entity.pose.quaternion_wxyz)}))
    solved = SceneSpec.model_validate(spec.model_dump() | {"entities": entities})
    validate_layout(solved, catalog)
    return solved
