# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from pathlib import Path

import trimesh

from .bundle import export_bundle
from .catalog import AssetCatalog
from .importers import import_robotwin
from .models import Entity, Goal, Physics, Pose, SceneSpec, Task, file_digest

ROBOTWIN_TASKS = ("mouse_pad", "stapler_pad", "pillbottle_pad", "object_scale", "stack_blocks")


def create_robotwin_task(name: str, source: Path, catalog: AssetCatalog, output: Path,
                        *, license: str, author: str) -> Path:
    if name not in ROBOTWIN_TASKS:
        raise ValueError(f"未知 RoboTwin 任务：{name}")
    objects = {
        "mouse_pad": ("047_mouse", 0, (0.045, 0.07, 0.025)),
        "stapler_pad": ("048_stapler", 0, (0.03, 0.065, 0.035)),
        "pillbottle_pad": ("080_pillbottle", 1, (0.025, 0.025, 0.045)),
        "object_scale": ("047_mouse", 0, (0.035, 0.055, 0.025)),
        "stack_blocks": ("086_woodenblock", 0, (0.03, 0.03, 0.03)),
    }
    directory, model_id, size = objects[name]
    obj = import_robotwin(catalog, source / directory, model_id, license=license, author=author)
    if name == "object_scale":
        target = import_robotwin(catalog, source / "072_electronicscale", 0, license=license, author=author)
        target_size = (0.10, 0.10, 0.025)
    elif name == "stack_blocks":
        target = obj
        target_size = (0.04, 0.04, 0.03)
    else:
        primitive = catalog.root / "primitives" / "pad.glb"
        primitive.parent.mkdir(parents=True, exist_ok=True)
        if not primitive.exists():
            trimesh.creation.box(extents=(1, 1, 1)).export(primitive)
        target = catalog.import_glb(primitive, uid=file_digest(primitive)[:32], metadata={
            "name": "task_pad", "viewerUrl": "procedural:trimesh.creation.box",
            "license": "MIT", "user": {"displayName": "OpenSO-101"},
        })
        target_size = (0.10, 0.10, 0.005)
    target_position = (0.25, -0.10, target_size[2] / 2)
    goal_position = (*target_position[:2], target_size[2] + size[2] / 2)
    spec = SceneSpec(
        scene_id=f"robotwin_{name}",
        entities=(
            Entity(entity_id="object", asset_uid=obj.uid, asset_sha256=obj.sha256,
                   dimensions_m=size, pose=Pose(position=(0.25, 0.08, size[2] / 2)),
                   physics=Physics(mass_kg=0.03),
                   reset_translation_m=((-0.01, -0.01, 0), (0.01, 0.01, 0))),
            Entity(entity_id="target", asset_uid=target.uid, asset_sha256=target.sha256,
                   dimensions_m=target_size, pose=Pose(position=target_position),
                   dynamic=name == "stack_blocks", physics=Physics(mass_kg=0.08)),
        ),
        task=Task(task_id=name, object_id="object", goal_position_m=goal_position,
                  instruction=f"将 {obj.name} 放置在 {target.name} 上面并释放夹爪。",
                  goals=(Goal(object_id="object", predicate="on_top", target_id="target"),),
                  position_tolerance_m=0.005),
    )
    return export_bundle(spec, catalog, output)
