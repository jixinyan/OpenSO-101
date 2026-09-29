# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
Positive = Annotated[float, Field(gt=0)]
Vector3 = tuple[float, float, float]
Dimensions = tuple[Positive, Positive, Positive]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class Pose(Model):
    position: Vector3 = (0.0, 0.0, 0.0)
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def unit_quaternion(self):
        if not math.isclose(sum(x * x for x in self.quaternion_wxyz), 1.0, abs_tol=1e-6):
            raise ValueError("quaternion_wxyz 必须为单位 quaternion")
        return self


class Asset(Model):
    uid: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    author: str = Field(min_length=1)
    sha256: Digest
    bounds: tuple[Vector3, Vector3]
    vertices: int = Field(gt=0)
    faces: int = Field(gt=0)
    format: Literal["glb", "usdz"] = "glb"

    @model_validator(mode="after")
    def valid_bounds(self):
        lower, upper = self.bounds
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("asset bounds 必须在每个轴上具有正尺寸")
        return self


class Physics(Model):
    mass_kg: Positive = 0.1
    static_friction: float = Field(default=0.8, ge=0)
    dynamic_friction: float = Field(default=0.6, ge=0)
    restitution: float = Field(default=0.0, ge=0, le=1)
    collision: Literal["convexHull", "convexDecomposition"] = "convexHull"
    provenance: Literal["user_provided", "measured", "estimated", "profile_default"] = "profile_default"

    @model_validator(mode="after")
    def friction_order(self):
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic_friction 不能超过 static_friction")
        return self


class Entity(Model):
    entity_id: Identifier
    asset_uid: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    asset_sha256: Digest
    dimensions_m: Dimensions
    pose: Pose
    dynamic: bool = True
    physics: Physics = Physics()
    reset_translation_m: tuple[Vector3, Vector3] = ((0, 0, 0), (0, 0, 0))

    @model_validator(mode="after")
    def reset_bounds(self):
        lower, upper = self.reset_translation_m
        if any(a > b for a, b in zip(lower, upper)):
            raise ValueError("reset_translation_m 下限必须小于或等于上限")
        return self


class Table(Model):
    center_xy_m: tuple[float, float] = (0.3, 0.0)
    size_xy_m: tuple[Positive, Positive] = (0.8, 0.8)
    top_z_m: float = 0.0
    thickness_m: Positive = 0.04


class Goal(Model):
    object_id: Identifier
    predicate: Literal["at", "inside", "on_top"] = "at"
    position_m: Vector3 | None = None
    target_id: Identifier | None = None
    region_bounds_m: tuple[Vector3, Vector3] | None = None

    @model_validator(mode="after")
    def goal_fields(self):
        if self.predicate == "at" and self.position_m is None:
            raise ValueError("at 目标需要 position_m")
        if self.predicate in ("inside", "on_top") and self.target_id is None:
            raise ValueError("关系目标需要 target_id")
        if self.predicate == "inside" and self.region_bounds_m is None:
            raise ValueError("inside 目标需要经过检查的容器内部区域")
        if self.region_bounds_m is not None and any(a >= b for a, b in zip(*self.region_bounds_m)):
            raise ValueError("容器内部区域必须具有正体积")
        return self


class Task(Model):
    task_id: Identifier
    object_id: Identifier
    goal_position_m: Vector3
    position_tolerance_m: Positive = 0.03
    settle_seconds: Positive = 0.5
    max_linear_speed_m_s: Positive = 0.02
    max_angular_speed_rad_s: Positive = 0.1
    instruction: str = ""
    goals: tuple[Goal, ...] = ()
    require_released: bool = True


class SceneSpec(Model):
    schema_version: Literal[1] = 1
    scene_id: Identifier
    meters_per_unit: Literal[1] = 1
    up_axis: Literal["Z"] = "Z"
    table: Table = Table()
    robot_base: Pose = Pose()
    entities: tuple[Entity, ...] = Field(min_length=1)
    task: Task
    reset_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unique_entities(self):
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity_id 必须唯一")
        target = next((entity for entity in self.entities if entity.entity_id == self.task.object_id), None)
        if target is None or not target.dynamic:
            raise ValueError("task.object_id 必须引用动态物体")
        for goal in self.task.goals:
            if goal.object_id == goal.target_id:
                raise ValueError("任务物体不能引用自身作为目标")
            if goal.object_id not in ids or (goal.target_id is not None and goal.target_id not in ids):
                raise ValueError("任务目标引用了不存在的实体")
            if not next(entity for entity in self.entities if entity.entity_id == goal.object_id).dynamic:
                raise ValueError("操作物体必须为动态物体")
        return self

    @classmethod
    def read(cls, path: Path) -> SceneSpec:
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls.model_validate_json(path.read_text())


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_document_digest(path: Path) -> str:
    document = json.loads(path.read_text())
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
