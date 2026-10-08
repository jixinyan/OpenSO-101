import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from openso101.scenes.bundle import export_bundle, verify_bundle
from openso101.scenes.capabilities import GeometryProbe, probe_geometry, verify_geometry_report
from openso101.scenes.catalog import AssetCatalog
from openso101.scenes.editing import EntityChange, SceneEdit, revise_bundle
from openso101.scenes.models import Entity, Goal, Pose, SceneSpec, Task, file_digest
from openso101.scenes.program import IntentAction, IntentObject, TaskCondition, TaskIntent, compile_program
from openso101.scenes.store import SceneStore


@pytest.fixture
def measured_scene(tmp_path):
    source = tmp_path / "block.glb"
    trimesh.creation.box(extents=(.04, .04, .04)).export(source)
    catalog = AssetCatalog(tmp_path / "catalog")
    asset = catalog.import_glb(source, uid="b" * 32, metadata={
        "name": "Measured block", "viewerUrl": "procedural:trimesh.creation.box",
        "license": "MIT", "user": {"displayName": "OpenSO-101"},
    })
    spec = SceneSpec(scene_id="measured_scene", entities=(Entity(
        entity_id="block", asset_uid=asset.uid, asset_sha256=asset.sha256,
        dimensions_m=(.04, .04, .04), pose=Pose(position=(.25, 0, .02))),),
        task=Task(task_id="place", object_id="block", goal_position_m=(.25, .15, .02)))
    return catalog, spec


def test_revision_preserves_unmodified_conditions(measured_scene, tmp_path):
    catalog, spec = measured_scene
    bundle = export_bundle(spec, catalog, tmp_path / "source")
    store = SceneStore(tmp_path / "scenes.sqlite")
    store.save(spec, expected_revision=0, reason="创建场景")
    edit = SceneEdit(scene_id=spec.scene_id, base_scene_sha256=spec.digest(), instruction="将物体向左移动两厘米",
                     changes=(EntityChange(entity_id="block", pose=Pose(position=(.25, -.02, .02))),))
    report = revise_bundle(bundle, edit, tmp_path / "revision", store=store, expected_revision=1)
    revised = verify_bundle(tmp_path / "revision")
    assert report["revision"] == 2
    assert revised.task == spec.task and revised.table == spec.table and revised.robot_base == spec.robot_base
    assert revised.entities[0].physics == spec.entities[0].physics
    assert store.read(spec.scene_id) == (2, revised)
    provenance = json.loads((tmp_path / "revision/provenance.json").read_text())
    assert provenance["base_scene_sha256"] == spec.digest()
    assert provenance["task_success_verified"] is False
    with pytest.raises(ValueError, match="版本已改变"):
        revise_bundle(bundle, edit, tmp_path / "stale", store=store, expected_revision=1)
    assert not (tmp_path / "stale").exists()


def test_colliding_edit_rejected_before_output(measured_scene, tmp_path):
    catalog, spec = measured_scene
    second = spec.entities[0].model_copy(update={"entity_id": "second", "pose": Pose(position=(.25, .1, .02))})
    spec = SceneSpec.model_validate(spec.model_dump() | {"entities": (*spec.entities, second)})
    source = export_bundle(spec, catalog, tmp_path / "source")
    edit = SceneEdit(scene_id=spec.scene_id, base_scene_sha256=spec.digest(), instruction="移动物体",
                     changes=(EntityChange(entity_id="block", pose=second.pose),))
    with pytest.raises(ValueError, match="检查未通过"):
        revise_bundle(source, edit, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def task_intent(spec):
    goal = Goal(object_id=spec.task.object_id, position_m=spec.task.goal_position_m)
    conditions = (TaskCondition(kind="goal", object_id=goal.object_id, goal=goal),
                  TaskCondition(kind="released", object_id=goal.object_id),
                  TaskCondition(kind="stable", object_id=goal.object_id))
    return TaskIntent(instruction=spec.task.instruction, objects=tuple(
        IntentObject(entity_id=entity.entity_id, role="manipulated", label="block", asset_query="block",
                     dimensions_m=entity.dimensions_m, initial_pose=entity.pose) for entity in spec.entities),
        ordered_actions=(IntentAction(action_id="place", operation="place", conditions=conditions),),
        final_conditions=conditions)


def test_goal_edit_updates_complete_program_and_keeps_other_conditions(measured_scene, tmp_path):
    catalog, spec = measured_scene
    spec = spec.model_copy(update={"task": spec.task.model_copy(update={"instruction": "将方块放到指定位置"})})
    intent = task_intent(spec)
    source = export_bundle(spec, catalog, tmp_path / "source", intent=intent, program=compile_program(intent, spec))
    goal = Goal(object_id="block", position_m=(.3, .15, .02))
    edit = SceneEdit(scene_id=spec.scene_id, base_scene_sha256=spec.digest(), instruction="移动放置目标",
                     goals=(goal,))
    revise_bundle(source, edit, tmp_path / "revision")
    revised = verify_bundle(tmp_path / "revision")
    revised_intent = TaskIntent.model_validate_json((tmp_path / "revision/task_intent.json").read_text())
    assert revised.entities == spec.entities
    assert revised_intent.final_conditions[0].goal == goal
    assert revised_intent.final_conditions[1:] == intent.final_conditions[1:]
    assert revised_intent.ordered_actions[0].conditions[0].goal == goal


def test_replacement_intent_accepts_complete_new_operation(measured_scene, tmp_path):
    catalog, spec = measured_scene
    spec = spec.model_copy(update={"task": spec.task.model_copy(update={"instruction": "放置方块"})})
    original = task_intent(spec)
    source = export_bundle(spec, catalog, tmp_path / "source", intent=original, program=compile_program(original, spec))
    revised_spec = spec.model_copy(update={"task": spec.task.model_copy(update={"instruction": "移动方块目标",
                                                                         "goal_position_m": (.3, .15, .02)})})
    replacement = task_intent(revised_spec)
    edit = SceneEdit(scene_id=spec.scene_id, base_scene_sha256=spec.digest(), instruction="修改任务",
                     goals=(replacement.final_conditions[0].goal,), task_instruction=replacement.instruction,
                     replacement_intent=replacement)
    revise_bundle(source, edit, tmp_path / "revision")
    assert verify_bundle(tmp_path / "revision").task.instruction == replacement.instruction
    assert TaskIntent.model_validate_json((tmp_path / "revision/task_intent.json").read_text()) == replacement


def test_asset_replacement_uses_external_catalog_and_portable_bundle(measured_scene, tmp_path):
    catalog, spec = measured_scene
    source = export_bundle(spec, catalog, tmp_path / "source")
    path = tmp_path / "replacement.glb"
    trimesh.creation.icosphere(radius=.02).export(path)
    replacement = catalog.import_glb(path, uid=file_digest(path)[:32], metadata={
        "name": "sphere", "viewerUrl": "procedural:trimesh.creation.icosphere", "license": "MIT",
        "user": {"displayName": "OpenSO-101"},
    })
    edit = SceneEdit(scene_id=spec.scene_id, base_scene_sha256=spec.digest(), instruction="替换操作物体的资产",
                     changes=(EntityChange(entity_id="block", asset_uid=replacement.uid),))
    revise_bundle(source, edit, tmp_path / "revision", catalog=catalog)
    revised = verify_bundle(tmp_path / "revision")
    assert revised.entities[0].asset_sha256 == replacement.sha256
    assert AssetCatalog(tmp_path / "revision/assets").read(replacement.uid) == replacement


def test_gripper_geometry_rejects_oversized_asset_and_changed_scope(measured_scene):
    catalog, spec = measured_scene
    probe = GeometryProbe(capability="graspable", dimensions_m=(.04, .04, .04),
                          maximum_gripper_opening_m=.05,
                          robot_sha256=file_digest(Path("outputs/so-arm100/Simulation/SO101/so101_old_calib.xml")),
                          collision="convexHull")
    report = probe_geometry(catalog, spec.entities[0].asset_uid, probe)
    assert report["geometry_accepted"] is True
    assert report["physical_task_verified"] is False
    verify_geometry_report(catalog, spec.entities[0].asset_uid, report, probe)
    enlarged = probe.model_copy(update={"dimensions_m": (.06, .06, .06)})
    assert probe_geometry(catalog, spec.entities[0].asset_uid, enlarged)["geometry_accepted"] is False
    with pytest.raises(ValueError, match="版本不一致"):
        verify_geometry_report(catalog, spec.entities[0].asset_uid, report, enlarged)


def test_container_measures_opening_walls_and_collision(tmp_path):
    # 旋转实际截面生成具有底部、壁厚与开口的封闭容器 mesh。
    mesh = trimesh.creation.revolve(np.array([(0, 0), (.06, 0), (.06, .1),
                                             (.05, .1), (.05, .01), (0, .01)]), sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    source = tmp_path / "container.glb"
    mesh.export(source)
    catalog = AssetCatalog(tmp_path / "catalog")
    asset = catalog.import_glb(source, uid="d" * 32, metadata={
        "name": "Measured container", "viewerUrl": "procedural:trimesh.creation.box",
        "license": "MIT", "user": {"displayName": "OpenSO-101"},
    })
    probe = GeometryProbe(capability="container", dimensions_m=(.12, .12, .10),
                          region_bounds_m=((-0.03, -.03, -.03), (.03, .03, .03)),
                          collision="convexDecomposition", samples_per_axis=3)
    report = probe_geometry(catalog, asset.uid, probe)
    assert report["geometry_accepted"] is True
    assert report["measurements"]["side_enclosed_samples"] == [27] * 4
    assert probe_geometry(catalog, asset.uid, probe.model_copy(update={"collision": "convexHull"}))[
        "geometry_accepted"] is False
    blocked = probe.model_copy(update={"region_bounds_m": ((-.03, -.03, -.049), (.03, .03, .03))})
    assert probe_geometry(catalog, asset.uid, blocked)["geometry_accepted"] is False
