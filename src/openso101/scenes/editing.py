import json
from pathlib import Path

from pydantic import Field, model_validator

from .bundle import export_bundle, verify_bundle
from .catalog import AssetCatalog
from .layout import diagnose_layout
from .models import Digest, Dimensions, Entity, Goal, Identifier, Model, Physics, Pose, SceneSpec, file_digest
from .program import IntentObject, TaskIntent, compile_program
from .store import SceneStore


class EntityChange(Model):
    entity_id: Identifier
    pose: Pose | None = None
    dimensions_m: Dimensions | None = None
    asset_uid: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    physics: Physics | None = None
    reset_translation_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    @model_validator(mode="after")
    def nonempty_change(self):
        if all(value is None for name, value in self.model_dump().items() if name != "entity_id"):
            raise ValueError("实体修改必须至少指定一个字段")
        return self


class SceneEdit(Model):
    scene_id: Identifier
    base_scene_sha256: Digest
    instruction: str = Field(min_length=1)
    changes: tuple[EntityChange, ...] = ()
    additions: tuple[Entity, ...] = ()
    removals: tuple[Identifier, ...] = ()
    goals: tuple[Goal, ...] | None = None
    task_instruction: str | None = Field(default=None, min_length=1)
    replacement_intent: TaskIntent | None = None

    @model_validator(mode="after")
    def unique_operations(self):
        changed = [item.entity_id for item in self.changes]
        added = [item.entity_id for item in self.additions]
        ids = [*changed, *added, *self.removals]
        if len(ids) != len(set(ids)):
            raise ValueError("同一实体只能具有一项修改操作")
        if not ids and self.goals is None and self.task_instruction is None and self.replacement_intent is None:
            raise ValueError("场景修改不能为空")
        if self.goals is not None and not self.goals:
            raise ValueError("场景目标不能为空")
        return self


def apply_scene_edit(spec: SceneSpec, edit: SceneEdit, catalog: AssetCatalog) -> SceneSpec:
    if edit.scene_id != spec.scene_id or edit.base_scene_sha256 != spec.digest():
        raise ValueError("场景修改与基础版本不一致")
    entities = {item.entity_id: item for item in spec.entities}
    for entity_id in edit.removals:
        if entity_id not in entities:
            raise ValueError(f"删除操作引用未知实体：{entity_id}")
        del entities[entity_id]
    for change in edit.changes:
        if change.entity_id not in entities:
            raise ValueError(f"修改操作引用未知实体：{change.entity_id}")
        values = change.model_dump(exclude_none=True, exclude={"entity_id"})
        if change.asset_uid is not None:
            values["asset_sha256"] = catalog.read(change.asset_uid).sha256
        entities[change.entity_id] = Entity.model_validate(entities[change.entity_id].model_dump() | values)
    for entity in edit.additions:
        if entity.entity_id in entities:
            raise ValueError(f"新增实体已经存在：{entity.entity_id}")
        if catalog.read(entity.asset_uid).sha256 != entity.asset_sha256:
            raise ValueError("新增实体的资产 SHA256 不一致")
        entities[entity.entity_id] = entity
    task = spec.task.model_dump()
    if edit.goals is not None:
        task.update(goals=[item.model_dump() for item in edit.goals])
        primary = next((item for item in edit.goals if item.object_id == spec.task.object_id), None)
        if primary is None:
            raise ValueError("修改目标需要保留主操作物体")
        if primary.predicate == "at":
            task["goal_position_m"] = primary.position_m
    if edit.task_instruction is not None:
        task["instruction"] = edit.task_instruction
    revised = SceneSpec.model_validate(spec.model_dump() | {"entities": list(entities.values()), "task": task})
    diagnostics = diagnose_layout(revised, catalog)
    if diagnostics["status"] != "static_checks_passed":
        raise ValueError(f"修改后的场景检查未通过：{diagnostics}")
    return revised


def revise_bundle(bundle: Path, edit: SceneEdit, output: Path, *, store: SceneStore | None = None,
                  expected_revision: int | None = None, model_requests=(), catalog: AssetCatalog | None = None) -> dict:
    spec = verify_bundle(bundle)
    if output.exists():
        raise FileExistsError(output)
    catalog = catalog if catalog is not None else AssetCatalog(bundle / "assets")
    revised = apply_scene_edit(spec, edit, catalog)
    intent = program = None
    if edit.replacement_intent is not None:
        intent = edit.replacement_intent
        program = compile_program(intent, revised)
    elif (bundle / "task_intent.json").is_file():
        original = TaskIntent.model_validate_json((bundle / "task_intent.json").read_text())
        if edit.additions or edit.removals:
            manipulated = {item.entity_id for item in original.objects if item.role != "decoration"}
            if manipulated.intersection(edit.removals):
                raise ValueError("删除操作物体需要完整的新 TaskIntent")
        objects = []
        revised_entities = {item.entity_id: item for item in revised.entities}
        for item in original.objects:
            if item.entity_id in edit.removals:
                continue
            entity = revised_entities[item.entity_id]
            values = item.model_dump()
            if original.schema_version == 2:
                values.update(dimensions_m=entity.dimensions_m, initial_pose=entity.pose.model_dump())
            if item.entity_id in {change.entity_id for change in edit.changes if change.asset_uid is not None}:
                asset = catalog.read(entity.asset_uid)
                values.update(label=asset.name, asset_query=asset.name)
            objects.append(values)
        for entity in edit.additions:
            asset = catalog.read(entity.asset_uid)
            objects.append(IntentObject(entity_id=entity.entity_id, label=asset.name, role="decoration",
                                        asset_query=asset.name, dimensions_m=entity.dimensions_m,
                                        initial_pose=entity.pose).model_dump())
        data = original.model_dump() | {"objects": objects, "instruction": revised.task.instruction}
        if edit.goals is not None:
            by_id = {goal.object_id: goal for goal in edit.goals}
            for action in data["ordered_actions"]:
                for condition in action["conditions"]:
                    if condition["kind"] == "goal" and condition["object_id"] in by_id:
                        condition["goal"] = by_id[condition["object_id"]].model_dump()
            for condition in data["final_conditions"]:
                if condition["kind"] == "goal":
                    condition["goal"] = by_id[condition["object_id"]].model_dump()
        intent = TaskIntent.model_validate(data)
        program = compile_program(intent, revised)
    if store is not None:
        current_revision, current = store.read(spec.scene_id)
        if expected_revision != current_revision or current.digest() != spec.digest():
            raise ValueError("保存的场景版本已改变")
    provenance = {"operation": "edit_scene", "base_scene_sha256": spec.digest(),
                  "base_manifest_sha256": file_digest(bundle / "manifest.json"),
                  "edit": edit.model_dump(mode="json"), "edit_sha256": edit.digest(),
                  "model_requests": list(model_requests), "source_sha256": file_digest(Path(__file__)),
                  "invalidated_checks": ["compilation", "physics", "reachability", "cameras", "collection"],
                  "task_success_verified": False, "dataset_verified": False}
    bddl_report = None
    if (bundle / "bddl.json").is_file():
        if edit.goals is not None or edit.replacement_intent is not None:
            raise ValueError("BDDL 任务目标修改需要重新绑定完整源 problem")
        bddl_report = json.loads((bundle / "bddl.json").read_text())
        bddl_report.update(scene_sha256=revised.digest(), task_success_verified=False,
                           initial_conditions_verified=False)
    export_bundle(revised, catalog, output, intent=intent, program=program, provenance=provenance,
                  bddl_report=bddl_report)
    revision = store.save(revised, expected_revision=expected_revision, reason=edit.instruction) if store else None
    return {"scene_id": spec.scene_id, "revision": revision, "base_scene_sha256": spec.digest(),
            "scene_sha256": revised.digest(), "bundle": str(output.resolve()),
            "status": "layout_valid", "pending_checks": provenance["invalidated_checks"]}


def interpret_scene_edit(bundle: Path, instruction: str, service, *, catalog: AssetCatalog | None = None) -> SceneEdit:
    spec = verify_bundle(bundle)
    catalog = catalog if catalog is not None else AssetCatalog(bundle / "assets")
    if not instruction.strip():
        raise ValueError("场景修改描述不能为空")
    edit = service.complete(
        system=("将用户修改转换为 SceneEdit，只填写用户要求修改的实体与字段。"
                "完整保留 scene_id、base_scene_sha256 和 instruction。"
                "未提及的实体、任务条件、机器人、桌面与物理参数保持当前配置。"
                "实体删除与新增必须由用户明确要求；新增实体必须提供目录中的资产 UID 和 SHA256。"
                "改变任务目标时提供完整 goals 与完整 task_instruction。"
                "任务操作物体变化时提供完整 replacement_intent。仅改变初始布局时保留任务文本。"),
        prompt=json.dumps({"instruction": instruction, "base_scene_sha256": spec.digest(),
                           "scene": spec.model_dump(mode="json"),
                           "assets": [asset.model_dump(mode="json") for asset in catalog.list()]},
                          ensure_ascii=False), schema=SceneEdit,
    )
    if edit.instruction != instruction:
        raise ValueError("场景修改必须保留用户原文")
    apply_scene_edit(spec, edit, catalog)
    return edit
