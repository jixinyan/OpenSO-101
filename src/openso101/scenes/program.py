import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .models import Digest, Dimensions, Goal, Identifier, Model, Pose, SceneSpec
from .task import evaluate_goals


class IntentObject(Model):
    entity_id: Identifier
    label: str = Field(min_length=1)
    role: Literal["manipulated", "target", "support", "decoration"]
    count: int = Field(default=1, ge=1)
    attributes: dict[str, str] = Field(default_factory=dict)
    required_capabilities: tuple[Literal["graspable", "container", "support_surface"], ...] = ()
    asset_query: str | None = Field(default=None, min_length=1)
    dimensions_m: Dimensions | None = None
    initial_pose: Pose | None = None


class TaskCondition(Model):
    kind: Literal["grasped", "released", "lifted", "goal", "stable"]
    object_id: Identifier
    goal: Goal | None = None
    height_above_table_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def condition_fields(self):
        if (self.kind == "goal") != (self.goal is not None):
            raise ValueError("goal 条件必须具有 Goal，其他条件不接受 Goal")
        if self.goal is not None and self.goal.object_id != self.object_id:
            raise ValueError("条件与 Goal 的 object_id 必须一致")
        if (self.kind == "lifted") != (self.height_above_table_m is not None):
            raise ValueError("lifted 条件必须具有高度，其他条件不接受高度")
        return self


class IntentAction(Model):
    action_id: Identifier
    operation: Literal["pick", "lift", "move", "place", "release", "wait"]
    conditions: tuple[TaskCondition, ...] = Field(min_length=1)
    hold_seconds: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def operation_conditions(self):
        required = {"pick": {"grasped"}, "lift": {"grasped", "lifted"},
                    "move": {"grasped", "goal"}, "place": {"goal", "released", "stable"},
                    "release": {"released"}, "wait": {"stable"}}[self.operation]
        if not required.issubset({item.kind for item in self.conditions}):
            raise ValueError(f"操作 {self.operation} 的检查条件不完整")
        return self


class TaskIntent(Model):
    schema_version: Literal[1, 2] = 2
    instruction: str = Field(min_length=1)
    objects: tuple[IntentObject, ...] = Field(min_length=1)
    ordered_actions: tuple[IntentAction, ...] = Field(min_length=1)
    final_conditions: tuple[TaskCondition, ...] = Field(min_length=1)
    constraints: tuple[str, ...] = ()
    confirmed_defaults: dict[str, str] = Field(default_factory=dict)
    clarifications: tuple[str, ...] = ()

    @model_validator(mode="after")
    def intent_references(self):
        for item in self.objects:
            geometry = (item.asset_query, item.dimensions_m, item.initial_pose)
            if self.schema_version == 1 and any(value is not None for value in geometry):
                raise ValueError("TaskIntent 版本 1 不接受版本 2 的几何字段")
            if self.schema_version == 2 and not self.clarifications and any(value is None for value in geometry):
                raise ValueError("TaskIntent 版本 2 需要检索文本、尺寸和初始 Pose")
        ids = [item.entity_id for item in self.objects]
        actions = [item.action_id for item in self.ordered_actions]
        if len(ids) != len(set(ids)) or len(actions) != len(set(actions)):
            raise ValueError("任务实体与动作的 identifier 必须唯一")
        for condition in (*self.final_conditions, *(item for action in self.ordered_actions for item in action.conditions)):
            if condition.object_id not in ids:
                raise ValueError("任务条件引用了未知物体")
            if condition.goal is not None and condition.goal.target_id is not None and condition.goal.target_id not in ids:
                raise ValueError("任务条件引用了未知目标")
        return self

    def digest(self):
        if self.schema_version == 1:
            excluded = {"objects": {"__all__": {"asset_query", "dimensions_m", "initial_pose"}}}
            return hashlib.sha256(self.model_dump_json(exclude=excluded).encode()).hexdigest()
        return super().digest()


class TaskProgram(Model):
    schema_version: Literal[1] = 1
    scene_sha256: Digest
    intent_sha256: Digest
    phases: tuple[IntentAction, ...] = Field(min_length=1)
    final_conditions: tuple[TaskCondition, ...] = Field(min_length=1)
    settle_seconds: float = Field(gt=0)
    grasp_force_threshold_newtons: float = Field(default=.5, gt=0)
    release_force_threshold_newtons: float = Field(default=.1, ge=0)

    def validate_scene(self, spec: SceneSpec):
        if self.scene_sha256 != spec.digest():
            raise ValueError("TaskProgram 与场景 SHA256 不一致")
        dynamic = {entity.entity_id for entity in spec.entities if entity.dynamic}
        ids = {entity.entity_id for entity in spec.entities}
        for condition in (*self.final_conditions, *(item for phase in self.phases for item in phase.conditions)):
            if condition.object_id not in dynamic:
                raise ValueError("操作条件必须引用动态实体")
            if condition.goal is not None and condition.goal.target_id is not None and condition.goal.target_id not in ids:
                raise ValueError("TaskProgram 引用了未知目标")
        if self.release_force_threshold_newtons >= self.grasp_force_threshold_newtons:
            raise ValueError("释放接触阈值必须小于抓取接触阈值")


def compile_program(intent: TaskIntent, spec: SceneSpec) -> TaskProgram:
    if intent.clarifications:
        raise ValueError("任务需要澄清：" + "；".join(intent.clarifications))
    if intent.instruction != spec.task.instruction:
        raise ValueError("场景必须保留 TaskIntent 的完整任务文本")
    ids = {entity.entity_id for entity in spec.entities}
    if any(item.count != 1 or item.entity_id not in ids for item in intent.objects):
        raise ValueError("每个任务实体需要展开为单独的 scene entity_id")
    entities = {item.entity_id: item for item in spec.entities}
    for item in intent.objects:
        entity = entities[item.entity_id]
        if item.role == "manipulated" and not entity.dynamic:
            raise ValueError("操作物体必须为动态实体")
        if intent.schema_version == 2:
            if (not np.allclose(item.dimensions_m, entity.dimensions_m, rtol=0, atol=1e-9)
                    or not np.allclose(item.initial_pose.position, entity.pose.position, rtol=0, atol=1e-9)
                    or abs(np.dot(item.initial_pose.quaternion_wxyz, entity.pose.quaternion_wxyz)) < 1 - 1e-8):
                raise ValueError(f"SceneSpec 必须保留任务实体的尺寸与初始 Pose：{item.entity_id}")
    goals = spec.task.goals or (Goal(object_id=spec.task.object_id, position_m=spec.task.goal_position_m),)
    final_goals = [item.goal for item in intent.final_conditions if item.kind == "goal"]
    if set(goal.digest() for goal in goals) != set(goal.digest() for goal in final_goals):
        raise ValueError("SceneSpec 与 TaskIntent 的全部最终目标必须一致")
    for goal in goals:
        required = {item.kind for item in intent.final_conditions if item.object_id == goal.object_id}
        if "stable" not in required or (spec.task.require_released and "released" not in required):
            raise ValueError("最终条件必须保留稳定与释放要求")
    program = TaskProgram(scene_sha256=spec.digest(), intent_sha256=intent.digest(),
                          phases=intent.ordered_actions, final_conditions=intent.final_conditions,
                          settle_seconds=spec.task.settle_seconds)
    program.validate_scene(spec)
    return program


def extract_intent(instruction, service):
    if not instruction.strip():
        raise ValueError("任务描述不能为空")
    intent = service.complete(
        system=("将用户任务转换为 schema_version=2 的 TaskIntent。完整保留 instruction，明确每个实体、数量、属性、"
                "操作顺序和最终条件。每个物体使用独立 entity_id，count=1。抓取使用 grasped，"
                "pick 必须包含 grasped；lift 包含 grasped 和 lifted；move 包含 grasped 和 goal；"
                "place 包含 goal、released 和 stable；release 包含 released；wait 包含 stable。"
                "每项 kind 对应单独的 TaskCondition。最终条件保留全部操作物体的 goal、released、stable。"
                "默认桌面属于 SceneSpec.table，使用 at 目标表示桌面放置，不创建 table 实体引用。"
                "每个实体提供英文 asset_query、米制 dimensions_m 与 initial_pose。用户提供的尺寸、"
                "位置和旋转完整保存到这些字段；未提供的参数采用 SO-101 桌面默认布局并记录到 confirmed_defaults。"
                "容器内部区域和其他缺少必要测量依据的信息填写 clarifications；"
                "不得编造测量结果或删除用户条件。"),
        prompt=instruction, schema=TaskIntent,
    )
    if intent.instruction != instruction or intent.schema_version != 2:
        raise ValueError("TaskIntent 必须保留完整任务文本")
    return intent


class TaskProgramTracker:
    def __init__(self, spec: SceneSpec, program: TaskProgram):
        program.validate_scene(spec)
        self.spec = spec
        self.program = program
        self.reset()

    def reset(self):
        self.phase = 0
        self.phase_hold_seconds = 0.
        self.final_hold_seconds = 0.

    def condition(self, condition, states, jaw_forces, gripper_open):
        state = np.asarray(states[condition.object_id])
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError("TaskProgram 需要实际完整实体状态")
        if condition.kind in ("grasped", "released"):
            forces = np.asarray(jaw_forces[condition.object_id])
            if forces.shape != (2,) or not np.isfinite(forces).all() or (forces < 0).any():
                raise ValueError("TaskProgram 需要实际双侧接触力")
            if condition.kind == "grasped":
                return bool((forces > self.program.grasp_force_threshold_newtons).all())
            return bool(gripper_open and (forces <= self.program.release_force_threshold_newtons).all())
        if condition.kind == "lifted":
            return bool(state[2] - self.spec.table.top_z_m >= condition.height_above_table_m)
        if condition.kind == "stable":
            return bool(np.linalg.norm(state[7:10]) <= self.spec.task.max_linear_speed_m_s
                        and np.linalg.norm(state[10:13]) <= self.spec.task.max_angular_speed_rad_s)
        task = self.spec.task.model_copy(update={"goals": (condition.goal,), "require_released": False})
        result = evaluate_goals(self.spec.model_copy(update={"task": task}), states, gripper_open)
        return result["conditions"][0]["reached"]

    def update(self, states, jaw_forces, gripper_open, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("TaskProgram dt 必须为正数有限数值")
        active = self.phase
        if active < len(self.program.phases):
            phase = self.program.phases[active]
            values = [self.condition(item, states, jaw_forces, gripper_open) for item in phase.conditions]
            self.phase_hold_seconds = self.phase_hold_seconds + dt if all(values) else 0.
            if all(values) and self.phase_hold_seconds + 1e-9 >= phase.hold_seconds:
                self.phase += 1
                self.phase_hold_seconds = 0.
        final = [self.condition(item, states, jaw_forces, gripper_open) for item in self.program.final_conditions]
        eligible = self.phase == len(self.program.phases) and all(final)
        self.final_hold_seconds = self.final_hold_seconds + dt if eligible else 0.
        return {"phase_before": active, "phase": self.phase, "phase_count": len(self.program.phases),
                "final_conditions": final, "held_seconds": self.final_hold_seconds,
                "success": self.final_hold_seconds + 1e-9 >= self.program.settle_seconds}


def read_program(path: Path, spec: SceneSpec):
    program = TaskProgram.model_validate_json(path.read_text())
    program.validate_scene(spec)
    return program
