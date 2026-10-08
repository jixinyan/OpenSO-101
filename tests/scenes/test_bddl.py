import json
import xml.etree.ElementTree as ET
from importlib.resources import files

import mujoco
import numpy as np
import pytest
import trimesh

from openso101.scenes.bddl import BDDLBinding, BDDLTaskTracker, bind_bddl_problem, read_bddl_problem
from openso101.scenes.catalog import AssetCatalog
from openso101.scenes.models import Entity, Pose, SceneSpec, Task, file_digest


def book_problem():
    return files("bddl").joinpath("activity_definitions/boxing_books_up_for_storage/problem0.bddl")


@pytest.fixture
def book_scene(tmp_path):
    # 使用 MuJoCo 的实际容器碰撞与动态物体状态验证完整量词目标。
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    world = ET.SubElement(root, "worldbody")
    container = ET.SubElement(world, "body", name="container", pos="0.3 0 0.05")
    walls = []
    for position, size in (
        ((0, 0, -.047), (.09, .09, .003)),
        ((-.087, 0, 0), (.003, .09, .05)), ((.087, 0, 0), (.003, .09, .05)),
        ((0, -.087, 0), (.09, .003, .05)), ((0, .087, 0), (.09, .003, .05)),
    ):
        ET.SubElement(container, "geom", type="box", pos=" ".join(map(str, position)),
                      size=" ".join(map(str, size)))
        wall = trimesh.creation.box(extents=2 * np.asarray(size))
        wall.apply_translation(position)
        walls.append(wall)
    catalog = AssetCatalog(tmp_path / "assets")
    assets = {}
    for name, mesh in (("container", trimesh.util.concatenate(walls)),
                       ("book", trimesh.creation.box(extents=(.02, .02, .02)))):
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
        source = tmp_path / f"{name}.glb"
        mesh.export(source)
        assets[name] = catalog.import_glb(source, uid=file_digest(source)[:32], metadata={
            "name": name, "viewerUrl": "procedural:trimesh.creation.box", "license": "MIT",
            "user": {"displayName": "OpenSO-101"},
        })
    entities = [Entity(entity_id="container", asset_uid=assets["container"].uid,
                       asset_sha256=assets["container"].sha256,
                       dimensions_m=(.18, .18, .1), pose=Pose(position=(.3, 0, .05)), dynamic=False)]
    mapping = {"box.n.01_1": "container"}
    for index in range(6):
        name = f"book_{index + 1}"
        position = (.26 + .04 * (index % 3), -.025 + .05 * (index // 3), .03)
        body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, position)))
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="box", size="0.01 0.01 0.01", mass="0.02")
        entities.append(Entity(entity_id=name, asset_uid=assets["book"].uid, asset_sha256=assets["book"].sha256,
                               dimensions_m=(.02, .02, .02), pose=Pose(position=position)))
        mapping[f"book.n.02_{index + 1}"] = name
    spec = SceneSpec(scene_id="book_logic", entities=tuple(entities),
                     task=Task(task_id="books", object_id="book_1", goal_position_m=(.26, -.025, .016)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1000)
    problem = read_bddl_problem(book_problem())
    binding = BDDLBinding(source_sha256=problem["source_sha256"], object_entities=mapping,
                          container_regions_m={"container": ((-.08, -.08, -.046), (.08, .08, .049))})
    return spec, model, data, problem, binding


def body_states(spec, model, data):
    result = {}
    for entity in spec.entities:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.entity_id)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        result[entity.entity_id] = np.concatenate((data.xpos[body], data.xquat[body], velocity[3:], velocity[:3]))
    return result


def test_official_problem_preserves_all_initial_and_quantified_goals():
    problem = read_bddl_problem(book_problem())
    assert len(problem["objects"]["book.n.02"]) == 6
    assert len(problem["initial_conditions"]) == 13
    assert problem["goal_conditions"][0][0] == "forall"
    assert problem["source_verified"] is True
    assert problem["initial_conditions_verified"] is False


def test_all_books_required_with_actual_mujoco_states(book_scene, tmp_path):
    spec, model, data, problem, binding = book_scene
    tracker = BDDLTaskTracker(problem, binding, spec)
    for _ in range(250):
        mujoco.mj_step(model, data)
        result = tracker.update(body_states(spec, model, data), True, model.opt.timestep)
    assert result["source_goal_satisfied"] is True and result["success"] is True
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "book_6")
    address = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[address] = .5
    mujoco.mj_forward(model, data)
    result = tracker.update(body_states(spec, model, data), True, model.opt.timestep)
    assert result["source_goal_satisfied"] is False and result["success"] is False
    assert result["held_seconds"] == 0
    report = bind_bddl_problem(book_problem(), binding, spec, tmp_path / "binding.json")
    assert report["initial_conditions_verified"] is False and report["task_success_verified"] is False
    assert json.loads((tmp_path / "binding.json").read_text())["problem"] == problem


def test_missing_quantifier_binding_and_region_are_rejected(book_scene):
    spec, _, _, problem, binding = book_scene
    incomplete = dict(binding.object_entities)
    del incomplete["book.n.02_6"]
    with pytest.raises(ValueError, match="全部对象"):
        BDDLTaskTracker(problem, binding.model_copy(update={"object_entities": incomplete}), spec)
    with pytest.raises(ValueError, match="内部区域"):
        BDDLTaskTracker(problem, binding.model_copy(update={"container_regions_m": {}}), spec)


def test_bddl_source_modification_is_rejected(book_scene):
    spec, _, _, problem, binding = book_scene
    with pytest.raises(ValueError, match="原文"):
        BDDLTaskTracker({**problem, "source_text": problem["source_text"] + "\n"}, binding, spec)
