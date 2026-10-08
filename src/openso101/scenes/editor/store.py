# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import hashlib
import json
import sqlite3
from pathlib import Path

from ..models import SceneSpec


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
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool) or expected_revision < 0:
            raise ValueError("版本必须为非负整数，修改原因不能为空")
        if not isinstance(reason, str) or not reason.strip():
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
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id 不能为空")
        if revision is not None and (not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0):
            raise ValueError("revision 必须为正整数")
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT revision, digest, document FROM scenes WHERE scene_id = ? "
                "AND (? IS NULL OR revision = ?) ORDER BY revision DESC LIMIT 1",
                (scene_id, revision, revision),
            ).fetchone()
        if row is None:
            raise KeyError((scene_id, revision))
        try:
            spec = SceneSpec.model_validate_json(row[2])
            document_digest = hashlib.sha256(
                json.dumps(json.loads(row[2]), ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("保存的场景文档无效") from exc
        if row[1] not in {spec.digest(), document_digest}:
            raise ValueError("保存的场景内容与 hash 不一致")
        return row[0], spec
