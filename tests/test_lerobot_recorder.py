import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_lerobot_recorder.py for the legacy assertions.",
    allow_module_level=True,
)

import numpy as np

from openso101.teleop.lerobot_recorder import (
    REQUIRED_CAMERA_NAMES,
    build_lerobot_features,
    collect_camera_buffers,
    discover_camera_metadata,
    ensure_required_cameras,
    has_lerobot_metadata,
    prepare_lerobot_root_for_create,
)


class FakeCameraData:
    def __init__(self, rgb):
        self.output = {"rgb": rgb}


class FakeCamera:
    def __init__(self, rgb, height=2, width=3):
        self.data = FakeCameraData(rgb)
        self.cfg = type("Cfg", (), {"height": height, "width": width})()


class FakeScene(dict):
    pass


class FakeInteractiveScene:
    def __init__(self, entities):
        self._entities = dict(entities)

    def keys(self):
        return self._entities.keys()

    def __getitem__(self, key):
        if key not in self._entities:
            raise KeyError(
                f"Scene entity with key '{key}' not found. "
                f"Available Entities: '{list(self._entities)}'"
            )
        return self._entities[key]


def test_features_include_wrist_and_overhead_video_streams():
    cameras = {
        "wrist_camera": {"height": 128, "width": 128},
        "overhead_camera": {"height": 128, "width": 128},
    }

    features = build_lerobot_features(cameras, fps=30)

    assert "observation.images.wrist_camera" in features
    assert "observation.images.overhead_camera" in features
    assert features["observation.images.wrist_camera"]["shape"] == (128, 128, 3)
    assert features["observation.images.overhead_camera"]["dtype"] == "video"


def test_required_camera_validation_reports_missing_camera():
    with pytest.raises(ValueError, match="overhead_camera"):
        ensure_required_cameras({"wrist_camera": {"height": 128, "width": 128}})


def test_collect_camera_buffers_extracts_rgb_uint8_and_drops_alpha():
    rgba = np.zeros((1, 2, 3, 4), dtype=np.uint8)
    rgba[..., 0] = 10
    rgba[..., 3] = 255
    scene = FakeScene(
        {
            "wrist_camera": FakeCamera(rgba),
            "overhead_camera": FakeCamera(rgba + 1),
        }
    )

    buffers = collect_camera_buffers(scene)

    assert set(buffers) == set(REQUIRED_CAMERA_NAMES)
    assert buffers["wrist_camera"].shape == (2, 3, 3)
    assert buffers["wrist_camera"].dtype == np.uint8
    assert np.all(buffers["wrist_camera"][..., 0] == 10)


def test_discover_camera_metadata_supports_isaac_interactive_scene_membership():
    scene = FakeInteractiveScene(
        {
            "wrist_camera": FakeCamera(np.zeros((1, 2, 3, 3), dtype=np.uint8), height=2, width=3),
            "overhead_camera": FakeCamera(np.zeros((1, 4, 5, 3), dtype=np.uint8), height=4, width=5),
        }
    )

    metadata = discover_camera_metadata(scene)

    assert metadata == {
        "wrist_camera": {"height": 2, "width": 3},
        "overhead_camera": {"height": 4, "width": 5},
    }


def test_collect_camera_buffers_supports_isaac_interactive_scene_membership():
    scene = FakeInteractiveScene(
        {
            "wrist_camera": FakeCamera(np.zeros((1, 2, 3, 3), dtype=np.uint8)),
            "overhead_camera": FakeCamera(np.ones((1, 2, 3, 3), dtype=np.uint8)),
        }
    )

    buffers = collect_camera_buffers(scene)

    assert buffers["wrist_camera"].shape == (2, 3, 3)
    assert buffers["overhead_camera"].shape == (2, 3, 3)


def test_prepare_lerobot_root_removes_only_empty_stale_directory(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    prepare_lerobot_root_for_create(root)

    assert not root.exists()


def test_prepare_lerobot_root_rejects_non_empty_invalid_directory(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "notes.txt").write_text("not a dataset")

    with pytest.raises(ValueError, match="not a LeRobot dataset"):
        prepare_lerobot_root_for_create(root)


def test_lerobot_metadata_requires_tasks_file_for_reopen(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")

    assert not has_lerobot_metadata(root)

    (meta / "tasks.parquet").write_bytes(b"")

    assert has_lerobot_metadata(root)


def test_prepare_lerobot_root_archives_incomplete_metadata_directory(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")

    archive = prepare_lerobot_root_for_create(root)

    assert archive is not None
    assert not root.exists()
    assert archive.name.startswith("dataset.incomplete-")
    assert (archive / "meta" / "info.json").is_file()


def test_prepare_lerobot_root_keeps_existing_lerobot_dataset(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    (meta / "tasks.parquet").write_bytes(b"")

    archive = prepare_lerobot_root_for_create(root)

    assert archive is None
    assert has_lerobot_metadata(root)
