# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

from .usd import verify_compilation


def store_recording_scene(compiled_scene: Path, dataset_root: Path) -> dict[str, str]:
    report = verify_compilation(compiled_scene)
    digest = report["scene_sha256"]
    relative = Path("scenes") / digest
    destination = dataset_root / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(compiled_scene, destination)
    if verify_compilation(destination)["scene_sha256"] != digest:
        raise ValueError("采集目录中的场景版本不匹配")
    return {"scene_sha256": digest, "scene_relative_path": relative.as_posix()}


def resolve_recording_scene(episode: Path, attrs) -> Path | None:
    if "scene_relative_path" not in attrs:
        if "scene_sha256" in attrs:
            raise ValueError("采集记录缺少场景路径")
        return None
    root = episode.resolve().parent.parent
    path = (root / str(attrs["scene_relative_path"])).resolve()
    if not path.is_relative_to(root):
        raise ValueError("场景路径超出采集目录")
    if verify_compilation(path)["scene_sha256"] != attrs["scene_sha256"]:
        raise ValueError("采集记录与场景版本不匹配")
    return path
