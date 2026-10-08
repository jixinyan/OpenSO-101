# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
import shutil

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from openso101.scenes.bundle import (
    entity_bounds,
    export_bundle,
    validate_layout,
    verify_bundle,
)
from openso101.scenes.assets.catalog import AssetCatalog
from openso101.scenes.layout import solve_layout
from openso101.scenes.models import Entity, Physics, Pose, SceneSpec, Task
from openso101.scenes.editor.store import SceneStore


@pytest.fixture
def setup_scene(tmp_path):
    # 使用有准确尺寸的实体 mesh 检查文件读取和几何计算。
    source = tmp_path / "box.glb"
    trimesh.creation.box(extents=(2, 3, 4)).export(source)
    catalog = AssetCatalog(tmp_path / "catalog")
    asset = catalog.import_glb(source, uid="a" * 32, metadata={
        "name": "Analytic box", "viewerUrl": "local:analytic-box",
        "license": "CC0", "user": {"displayName": "OpenSO-101 tests"},
    })
    spec = SceneSpec(scene_id="box_scene", entities=(Entity(
        entity_id="object", asset_uid=asset.uid, asset_sha256=asset.sha256,
        dimensions_m=(0.04, 0.06, 0.08), pose=Pose(position=(0.25, 0, 0.04)),
    ),), task=Task(task_id="place", object_id="object", goal_position_m=(0.25, 0.15, 0.04)))
    return catalog, spec


def test_asset_dimensions_and_cache(setup_scene):
    catalog, spec = setup_scene
    asset = catalog.read(spec.entities[0].asset_uid)
    np.testing.assert_allclose(np.diff(asset.bounds, axis=0)[0], (2, 3, 4))
    assert catalog.list() == [asset]
    assert catalog.fetch(asset.uid) == asset


def test_scene_revisions_reject_stale_edits(setup_scene, tmp_path):
    _, spec = setup_scene
    path = tmp_path / "scenes.sqlite"
    store = SceneStore(path)
    assert store.save(spec, expected_revision=0, reason="创建场景") == 1
    edited = spec.model_copy(update={"reset_seed": 7})
    assert SceneStore(path).save(edited, expected_revision=1, reason="修改随机种子") == 2
    with pytest.raises(ValueError, match="场景版本已改变"):
        store.save(spec, expected_revision=1, reason="过期修改")
    assert store.read(spec.scene_id) == (2, edited)
    assert store.read(spec.scene_id, 1) == (1, spec)


def test_layout_separates_objects_and_preserves_locked_pose(setup_scene):
    catalog, spec = setup_scene
    second = spec.entities[0].model_copy(update={"entity_id": "second"})
    overlapping = SceneSpec.model_validate(spec.model_dump() | {"entities": (*spec.entities, second)})
    result = solve_layout(overlapping, catalog, locked=("object",), clearance_m=0.02)
    assert result.entities[0].pose == spec.entities[0].pose
    first_bounds, second_bounds = map(entity_bounds, result.entities)
    gap = np.maximum(first_bounds[0, :2] - second_bounds[1, :2], second_bounds[0, :2] - first_bounds[1, :2])
    assert gap.max() >= 0.02 - 1e-7
    with pytest.raises(ValueError, match="布局求解失败"):
        solve_layout(overlapping, catalog, locked=("object", "second"))


def test_portable_bundle_without_original_files(setup_scene, tmp_path):
    catalog, spec = setup_scene
    bundle = export_bundle(spec, catalog, tmp_path / "bundle")
    moved = tmp_path / "moved"
    shutil.move(bundle, moved)
    shutil.rmtree(catalog.root)
    assert verify_bundle(moved) == spec
    assert json.loads((moved / "validation.json").read_text())["status"] == "layout_valid"


def test_bundle_does_not_overwrite(setup_scene, tmp_path):
    catalog, spec = setup_scene
    bundle = export_bundle(spec, catalog, tmp_path / "bundle")
    with pytest.raises(FileExistsError):
        export_bundle(spec, catalog, bundle)
    assert verify_bundle(bundle) == spec


def test_modified_asset_rejected(setup_scene, tmp_path):
    catalog, spec = setup_scene
    bundle = export_bundle(spec, catalog, tmp_path / "bundle")
    path = bundle / "assets" / spec.entities[0].asset_uid / "model.glb"
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="校验失败"):
        verify_bundle(bundle)


def test_incomplete_manifest_rejected(setup_scene, tmp_path):
    catalog, spec = setup_scene
    bundle = export_bundle(spec, catalog, tmp_path / "bundle")
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["scene.json"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="必需文件"):
        verify_bundle(bundle)


@pytest.mark.parametrize("position", [(0.25, 0, 0.01), (0.99, 0, 0.04)])
def test_invalid_placement_rejected(setup_scene, position):
    catalog, spec = setup_scene
    data = spec.model_dump()
    data["entities"][0]["pose"]["position"] = position
    with pytest.raises(ValueError, match="桌面"):
        validate_layout(SceneSpec.model_validate(data), catalog)


def test_reset_envelope_rejected(setup_scene):
    catalog, spec = setup_scene
    data = spec.model_dump()
    data["entities"][0]["reset_translation_m"] = ((0, 0, -0.01), (0, 0, 0))
    with pytest.raises(ValueError, match="reset"):
        validate_layout(SceneSpec.model_validate(data), catalog)


def test_rotated_bounds(setup_scene):
    _, spec = setup_scene
    data = spec.entities[0].model_dump()
    data["pose"]["quaternion_wxyz"] = (2 ** -0.5, 0, 0, 2 ** -0.5)
    bounds = entity_bounds(Entity.model_validate(data))
    np.testing.assert_allclose(bounds[1] - bounds[0], (0.06, 0.04, 0.08), atol=1e-10)


@pytest.mark.parametrize("quaternion", [(0, 0, 0, 0), (2, 0, 0, 0), (float("nan"), 0, 0, 0)])
def test_invalid_rotation_rejected(quaternion):
    with pytest.raises(ValidationError):
        Pose(quaternion_wxyz=quaternion)


def test_nonfinite_position_rejected():
    with pytest.raises(ValidationError):
        Pose(position=(float("inf"), 0, 0))


def test_invalid_mass_and_friction_rejected():
    with pytest.raises(ValidationError):
        Physics(mass_kg=0)
    with pytest.raises(ValidationError):
        Physics(static_friction=0.1, dynamic_friction=0.5)


def test_task_reference_rejected(setup_scene):
    _, spec = setup_scene
    data = spec.model_dump()
    data["task"]["object_id"] = "missing"
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(data)


def test_path_traversal_rejected(tmp_path):
    catalog = AssetCatalog(tmp_path / "catalog")
    with pytest.raises(ValueError):
        catalog.read("../outside")


def test_unknown_fields_rejected(setup_scene):
    _, spec = setup_scene
    data = spec.model_dump()
    data["mass"] = 3
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(data)
