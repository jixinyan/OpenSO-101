import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_lerobot_push_dataset.py for the legacy assertions.",
    allow_module_level=True,
)

import h5py
import numpy as np

from openso101.scripts.lerobot.push_dataset import _detect_input_format, _validate_local_dataset


def test_push_dataset_rejects_incomplete_local_lerobot_metadata(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")

    with pytest.raises(SystemExit, match="Missing LeRobot metadata"):
        _validate_local_dataset(root, input_format="lerobot")


def test_push_dataset_requires_recorded_lerobot_episode_files(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    (meta / "tasks.parquet").write_bytes(b"")
    (meta / "stats.json").write_text("{}")

    with pytest.raises(SystemExit, match="no recorded episode"):
        _validate_local_dataset(root, input_format="lerobot")


def test_push_dataset_accepts_complete_local_lerobot_episode_layout(tmp_path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    (meta / "tasks.parquet").write_bytes(b"")
    (meta / "stats.json").write_text("{}")
    episode_file = root / "data" / "chunk-000" / "file-000.parquet"
    episode_file.parent.mkdir(parents=True)
    episode_file.write_bytes(b"")

    assert _validate_local_dataset(root, input_format="lerobot") == [episode_file]


def test_push_dataset_detects_hdf5_teleop_dataset(tmp_path):
    root = tmp_path / "teleop_data"
    episode = root / "episodes" / "episode_000000.hdf5"
    episode.parent.mkdir(parents=True)
    with h5py.File(episode, "w") as h5:
        h5.create_dataset("action", data=np.zeros((1, 6), dtype=np.float32))
        observations = h5.create_group("observations")
        observations.create_dataset("qpos", data=np.zeros((1, 6), dtype=np.float32))
        observations.create_dataset("qvel", data=np.zeros((1, 6), dtype=np.float32))
        images = observations.create_group("images")
        images.create_dataset("wrist_camera", data=np.zeros((1, 16, 16, 3), dtype=np.uint8))
        images.create_dataset("overhead_camera", data=np.zeros((1, 16, 16, 3), dtype=np.uint8))
        h5.create_dataset("timestamps", data=np.zeros((1,), dtype=np.float64))

    assert _detect_input_format(root) == "hdf5"
    assert _validate_local_dataset(root, input_format="hdf5") == [episode]
