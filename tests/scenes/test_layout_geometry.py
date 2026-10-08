import numpy as np
import pytest
import trimesh

from openso101.scenes.catalog import AssetCatalog
from openso101.scenes.layout import diagnose_layout, solve_layout
from openso101.scenes.models import Entity, Physics, Pose, SceneSpec, Task, file_digest


@pytest.fixture
def container_scene(tmp_path):
    catalog = AssetCatalog(tmp_path / "assets")
    container = trimesh.creation.revolve(np.array([(0, 0), (.06, 0), (.06, .1),
                                                  (.05, .1), (.05, .01), (0, .01)]), sections=64)
    container.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    block = trimesh.creation.box(extents=(.03, .03, .03))
    assets = []
    for name, mesh in (("container", container), ("block", block)):
        source = tmp_path / f"{name}.glb"
        mesh.export(source)
        assets.append(catalog.import_glb(source, uid=file_digest(source)[:32], metadata={
            "name": name, "viewerUrl": "procedural:trimesh.creation", "license": "MIT",
            "user": {"displayName": "OpenSO-101"}}))
    container_asset, block_asset = assets
    spec = SceneSpec(scene_id="container_scene", entities=(
        Entity(entity_id="container", asset_uid=container_asset.uid, asset_sha256=container_asset.sha256,
               dimensions_m=(.12, .12, .1), pose=Pose(position=(.25, 0, .05)), dynamic=False,
               physics=Physics(collision="convexDecomposition")),
        Entity(entity_id="block", asset_uid=block_asset.uid, asset_sha256=block_asset.sha256,
               dimensions_m=(.03, .03, .03), pose=Pose(position=(.25, 0, .05)))),
        task=Task(task_id="place", object_id="block", goal_position_m=(.25, .2, .015)))
    return catalog, spec


def test_actual_container_cavity_keeps_initial_layout(container_scene):
    catalog, spec = container_scene
    report = diagnose_layout(spec, catalog)
    assert report["status"] == "static_checks_passed"
    assert report["collisions"] == [] and report["physics_collision_cooking_verified"] is False
    assert solve_layout(spec, catalog, locked=("container", "block")) == spec


@pytest.mark.parametrize("position", [(.30, 0, .05), (.25, 0, .015)])
def test_actual_container_wall_and_bottom_intersection(container_scene, position):
    catalog, spec = container_scene
    block = spec.entities[1].model_copy(update={"pose": Pose(position=position)})
    revised = spec.model_copy(update={"entities": (spec.entities[0], block)})
    report = diagnose_layout(revised, catalog)
    assert report["status"] == "static_checks_failed"
    assert report["collisions"] == [["container", "block"]]


def test_convex_hull_contains_object_volume(container_scene):
    catalog, spec = container_scene
    solid = spec.entities[0].model_copy(update={"physics": Physics(collision="convexHull")})
    report = diagnose_layout(spec.model_copy(update={"entities": (solid, spec.entities[1])}), catalog)
    assert report["collisions"] == [["container", "block"]]
