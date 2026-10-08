# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import itertools

import numpy as np
import trimesh
from scipy.optimize import Bounds, LinearConstraint, milp

from .bundle import entity_bounds, validate_layout
from .catalog import AssetCatalog
from .capabilities import instance_mesh
from .models import Pose, SceneSpec


def sample_reset_positions(
    spec: SceneSpec,
    count: int,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample deterministic reset translations from the scene bounds.

    The runtime sampler lives in Isaac Lab and uses device tensors.  This
    CPU helper makes the same contract available to scene tooling and CI:
    every returned position is the entity pose plus a uniformly sampled
    translation inside ``reset_translation_m``.  Static entities are
    returned at their configured pose for convenient complete-state checks.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    rng = np.random.default_rng(spec.reset_seed if seed is None else seed)
    samples: dict[str, np.ndarray] = {}
    for entity in spec.entities:
        if entity.dynamic:
            lower, upper = np.asarray(entity.reset_translation_m, dtype=float)
            translation = rng.uniform(lower, upper, size=(count, 3))
            samples[entity.entity_id] = np.asarray(entity.pose.position, dtype=float) + translation
        else:
            samples[entity.entity_id] = np.broadcast_to(
                np.asarray(entity.pose.position, dtype=float), (count, 3)
            ).copy()
    return samples


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
    current_fits = all(np.all(shape[0, :2] + entity.pose.position[:2] >= table_lower)
                       and np.all(shape[1, :2] + entity.pose.position[:2] <= table_upper)
                       and shape[0, 2] + entity.pose.position[2] >= spec.table.top_z_m - 1e-6
                       for shape, entity in zip(reset_shapes, spec.entities))
    if current_fits and diagnose_layout(spec, catalog, clearance_m=clearance_m)["status"] == "static_checks_passed":
        return spec
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
        entities.append(
            entity.model_copy(
                update={"pose": Pose(position=position, quaternion_wxyz=entity.pose.quaternion_wxyz)}
            )
        )
    solved = SceneSpec.model_validate(spec.model_dump() | {"entities": entities})
    if diagnose_layout(solved, catalog)["status"] != "static_checks_passed":
        raise ValueError("求解位置未通过实际 mesh 碰撞检查")
    return solved


def _collision_mesh(entity, catalog):
    mesh = instance_mesh(catalog, entity.asset_uid, entity.dimensions_m,
                         require_volume=entity.physics.collision == "convexDecomposition")
    if entity.physics.collision == "convexHull":
        mesh = mesh.convex_hull
    transform = trimesh.transformations.quaternion_matrix(entity.pose.quaternion_wxyz)
    transform[:3, 3] = entity.pose.position
    mesh.apply_transform(transform)
    return mesh


def _intersects_material(first, second):
    manager = trimesh.collision.CollisionManager()
    manager.add_object("first", first)
    return bool(manager.in_collision_single(second) or first.contains(second.vertices).any()
                or second.contains(first.vertices).any())


def diagnose_layout(
    spec: SceneSpec,
    catalog: AssetCatalog,
    *,
    clearance_m: float = 0.0,
) -> dict:
    if clearance_m < 0 or not np.isfinite(clearance_m):
        raise ValueError("clearance_m must be a non-negative finite number")
    validate_layout(spec, catalog)
    bounds = {entity.entity_id: entity_bounds(entity) for entity in spec.entities}
    collisions: list[tuple[str, str]] = []
    meshes = {}
    for first, second in itertools.combinations(spec.entities, 2):
        a, b = bounds[first.entity_id], bounds[second.entity_id]
        overlap = np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])
        if np.all(overlap > clearance_m):
            for entity in (first, second):
                if entity.entity_id not in meshes:
                    meshes[entity.entity_id] = _collision_mesh(entity, catalog)
            if _intersects_material(meshes[first.entity_id], meshes[second.entity_id]):
                collisions.append((first.entity_id, second.entity_id))

    relation_errors: list[str] = []
    entities = {entity.entity_id: entity for entity in spec.entities}
    goals = spec.task.goals
    for goal in goals:
        if goal.predicate == "inside":
            lower, upper = np.asarray(goal.region_bounds_m, dtype=float)
            size = np.asarray(entities[goal.object_id].dimensions_m, dtype=float)
            if np.any(size > upper - lower + 1e-9):
                relation_errors.append(
                    f"inside goal {goal.object_id}->{goal.target_id} region is smaller than object"
                )
        elif goal.predicate == "on_top":
            object_size = np.asarray(entities[goal.object_id].dimensions_m, dtype=float)
            target_size = np.asarray(entities[goal.target_id].dimensions_m, dtype=float)
            if np.any(object_size[:2] > target_size[:2] + 2 * spec.task.position_tolerance_m):
                relation_errors.append(
                    f"on_top goal {goal.object_id}->{goal.target_id} target footprint is too small"
                )
    return {
        "status": "static_checks_passed" if not collisions and not relation_errors else "static_checks_failed",
        "collisions": [list(pair) for pair in collisions],
        "collision_scope": "instance_mesh_and_configured_convex_hulls",
        "physics_collision_cooking_verified": False,
        "relation_errors": relation_errors,
        "pending_checks": ["stability", "reachability", "contact_geometry", "cameras", "collection"],
    }
