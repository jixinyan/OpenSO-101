import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from bddl.activity import Conditions, evaluate_goal_conditions, get_goal_conditions, get_object_scope
from bddl.backend_abc import BDDLBackend
from bddl.logic_base import BinaryAtomicFormula
from bddl.parsing import parse_problem
from pydantic import Field

from .models import Digest, Goal, Identifier, Model, SceneSpec, Vector3, file_digest
from .task import evaluate_goals


class BDDLBinding(Model):
    source_sha256: Digest
    object_entities: dict[str, Identifier]
    container_regions_m: dict[Identifier, tuple[Vector3, Vector3]] = Field(default_factory=dict)


class BDDLProgress(Model):
    schema_version: Literal[1] = 1
    source_sha256: Digest
    binding_sha256: Digest
    scene_sha256: Digest
    held_seconds: float = Field(ge=0)


def read_bddl_problem(source: Path) -> dict:
    text = source.read_text()
    name, objects, initial, goals = parse_problem(source.stem, 0, "omnigibson", predefined_problem=text)
    if not objects or not initial or not goals:
        raise ValueError("BDDL problem 需要对象、初始条件和目标条件")
    return {"schema_version": 1, "problem": name, "objects": objects, "initial_conditions": initial,
            "goal_conditions": goals, "source_sha256": file_digest(source), "source_text": text,
            "parser": "bddl.parsing.parse_problem", "bddl_version": version("bddl"),
            "source_verified": True, "initial_conditions_verified": False,
            "task_success_verified": False}


class SceneObject:
    def __init__(self, tracker, entity_id):
        self.tracker = tracker
        self.entity_id = entity_id

    def relation(self, predicate, target):
        if target.tracker is not self.tracker:
            raise ValueError("BDDL 对象需要属于同一场景")
        tracker = self.tracker
        goal = Goal(object_id=self.entity_id, predicate=predicate, target_id=target.entity_id,
                    region_bounds_m=tracker.binding.container_regions_m.get(target.entity_id)
                    if predicate == "inside" else None)
        spec = tracker.spec.model_copy(update={"task": tracker.spec.task.model_copy(
            update={"goals": (goal,), "require_released": False})})
        return evaluate_goals(spec, tracker.states, True)["conditions"][0]["reached"]


class OnTop(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        return obj1.relation("on_top", obj2)

    def _sample(self, obj1, obj2, binary_state):
        raise ValueError("BDDL 初始布局需要独立的场景求解与验证")


class Inside(OnTop):
    STATE_NAME = "inside"

    def _evaluate(self, obj1, obj2):
        return obj1.relation("inside", obj2)


class SceneBDDLBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        predicates = {"ontop": OnTop, "inside": Inside}
        if predicate_name not in predicates:
            raise ValueError(f"当前场景不支持 BDDL predicate：{predicate_name}")
        return predicates[predicate_name]


class BDDLTaskTracker:
    def __init__(self, problem: dict, binding: BDDLBinding, spec: SceneSpec):
        if binding.source_sha256 != problem["source_sha256"]:
            raise ValueError("BDDL binding 与原始 problem SHA256 不一致")
        import hashlib

        if hashlib.sha256(problem["source_text"].encode()).hexdigest() != problem["source_sha256"]:
            raise ValueError("BDDL 原文与 SHA256 不一致")
        self.problem = problem
        self.binding = binding
        self.spec = spec
        self.states = None
        self.elapsed = 0.
        conditions = Conditions(problem["problem"], 0, "omnigibson", predefined_problem=problem["source_text"])
        scope = get_object_scope(conditions)
        entity_ids = {entity.entity_id for entity in spec.entities}
        if set(binding.object_entities) - set(scope) or set(binding.object_entities.values()) - entity_ids:
            raise ValueError("BDDL binding 引用了未知对象或场景实体")
        if len(set(binding.object_entities.values())) != len(binding.object_entities):
            raise ValueError("不同 BDDL 对象必须绑定不同场景实体")
        for name, entity_id in binding.object_entities.items():
            scope[name] = SceneObject(self, entity_id)
        self.conditions = get_goal_conditions(conditions, SceneBDDLBackend(), scope, generate_ground_options=False)
        self._validate_bindings(self.conditions)
        self.subject_ids = set().union(*(self._subjects(expression) for expression in self.conditions))
        if not self.subject_ids:
            raise ValueError("BDDL 目标需要实际操作物体")
        dynamic = {entity.entity_id for entity in spec.entities if entity.dynamic}
        if not self.subject_ids.issubset(dynamic):
            raise ValueError("BDDL 操作物体必须为动态实体")

    def _validate_bindings(self, expressions):
        for expression in expressions:
            if isinstance(expression, BinaryAtomicFormula):
                first, second = expression.scope[expression.input1], expression.scope[expression.input2]
                if not isinstance(first, SceneObject) or not isinstance(second, SceneObject):
                    raise ValueError("BDDL 目标中的全部对象需要明确的场景绑定")
                if expression.STATE_NAME == "inside" and second.entity_id not in self.binding.container_regions_m:
                    raise ValueError("BDDL inside 目标需要容器内部区域证据")
            self._validate_bindings(expression.children)

    def reset(self):
        self.elapsed = 0.
        self.states = None

    def snapshot(self):
        return BDDLProgress(source_sha256=self.problem["source_sha256"], binding_sha256=self.binding.digest(),
                            scene_sha256=self.spec.digest(), held_seconds=self.elapsed)

    def restore(self, progress: BDDLProgress):
        progress = BDDLProgress.model_validate(progress.model_dump() if isinstance(progress, BDDLProgress) else progress)
        current = self.snapshot()
        if (progress.source_sha256, progress.binding_sha256, progress.scene_sha256) != (
                current.source_sha256, current.binding_sha256, current.scene_sha256):
            raise ValueError("BDDL checkpoint 的来源、绑定与场景 SHA256 不一致")
        self.elapsed = progress.held_seconds
        self.states = None

    def update(self, states, gripper_open, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("BDDL dt 必须为正数有限数值")
        self.states = states
        # 复用实际实体状态检查，BDDL 库负责量词、逻辑关系与完整目标判定。
        evaluate_goals(self.spec, states, gripper_open)
        reached, details = evaluate_goal_conditions(self.conditions)
        stable = all(np.linalg.norm(states[name][7:10]) <= self.spec.task.max_linear_speed_m_s
                     and np.linalg.norm(states[name][10:13]) <= self.spec.task.max_angular_speed_rad_s
                     for name in self.subject_ids)
        eligible = reached and stable and (gripper_open or not self.spec.task.require_released)
        self.elapsed = self.elapsed + dt if eligible else 0.
        return {"source_goal_satisfied": bool(reached), "details": details,
                "stable": bool(stable), "held_seconds": self.elapsed,
                "success": bool(eligible and self.elapsed + 1e-9 >= self.spec.task.settle_seconds)}

    def _subjects(self, expression):
        if isinstance(expression, BinaryAtomicFormula):
            return {expression.scope[expression.input1].entity_id}
        return set().union(*(self._subjects(child) for child in expression.children))


def bind_bddl_problem(source: Path, binding: BDDLBinding, spec: SceneSpec, output: Path):
    problem = read_bddl_problem(source)
    BDDLTaskTracker(problem, binding, spec)
    report = {"problem": problem, "binding": binding.model_dump(mode="json"),
              "scene_sha256": spec.digest(), "source_sha256": file_digest(Path(__file__)),
              "status": "bddl_goal_bound", "initial_conditions_verified": False,
              "task_success_verified": False,
              "pending_checks": ["initial_conditions", "container_geometry", "physical_task", "collection"]}
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report
