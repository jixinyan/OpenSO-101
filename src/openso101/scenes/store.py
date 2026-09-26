# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import sqlite3
from pathlib import Path

from .models import SceneSpec


class SceneStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    document TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(scene_id, revision)
                )
            """)

    def save(self, spec: SceneSpec, *, expected_revision: int, reason: str) -> int:
        if expected_revision < 0 or not reason.strip():
            raise ValueError("版本必须为非负整数，修改原因不能为空")
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT MAX(revision) FROM scenes WHERE scene_id = ?", (spec.scene_id,)).fetchone()
            current = row[0] or 0
            if current != expected_revision:
                raise ValueError(f"场景版本已改变：expected={expected_revision}, current={current}")
            revision = current + 1
            connection.execute("INSERT INTO scenes VALUES (?, ?, ?, ?, ?)",
                               (spec.scene_id, revision, spec.digest(), spec.model_dump_json(), reason))
        return revision

    def read(self, scene_id: str, revision: int | None = None) -> tuple[int, SceneSpec]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT revision, digest, document FROM scenes WHERE scene_id = ? "
                "AND (? IS NULL OR revision = ?) ORDER BY revision DESC LIMIT 1",
                (scene_id, revision, revision),
            ).fetchone()
        if row is None:
            raise KeyError((scene_id, revision))
        spec = SceneSpec.model_validate_json(row[2])
        if spec.digest() != row[1]:
            raise ValueError("保存的场景内容与 hash 不一致")
        return row[0], spec
