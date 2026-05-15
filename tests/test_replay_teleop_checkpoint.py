import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_replay_teleop_checkpoint.py for the legacy assertions.",
    allow_module_level=True,
)

import h5py
import numpy as np

from openso101.scripts.lerobot.replay_teleop_checkpoint import (
    read_checkpoint_frames,
    replay_frame_range,
    resolve_episode_path,
    select_checkpoint_frame,
)
from openso101.teleop.hdf5_recorder import SafeSim2RealHDF5TeleopRecorder


CAMERAS = {
    "wrist_camera": {"height": 2, "width": 3},
    "overhead_camera": {"height": 2, "width": 3},
}


def _write_episode(root, frame_count=4):
    recorder = SafeSim2RealHDF5TeleopRecorder(root=root, task_name="task", cameras=CAMERAS, fps=30)
    recorder.start_episode()
    for frame_index in range(frame_count):
        recorder.add_frame(
            action=np.full(6, frame_index, dtype=np.float32),
            qpos=np.full(6, frame_index + 0.1, dtype=np.float32),
            qvel=np.full(6, frame_index + 0.2, dtype=np.float32),
            camera_buffers={
                "wrist_camera": np.zeros((2, 3, 3), dtype=np.uint8),
                "overhead_camera": np.zeros((2, 3, 3), dtype=np.uint8),
            },
            timestamp=float(frame_index),
        )
        if frame_index in {1, 3}:
            recorder.create_checkpoint()
    return recorder.save_episode()


def test_resolve_episode_path_uses_latest_episode_by_default(tmp_path):
    root = tmp_path / "teleop_data"
    first = _write_episode(root)
    second = _write_episode(root)

    assert resolve_episode_path(root) == second.resolve()
    assert resolve_episode_path(root, episode_index=0) == first.resolve()


def test_select_checkpoint_frame_uses_saved_checkpoint_index(tmp_path):
    episode = _write_episode(tmp_path / "teleop_data")

    assert read_checkpoint_frames(episode).tolist() == [1, 3]
    assert select_checkpoint_frame(episode) == 3
    assert select_checkpoint_frame(episode, checkpoint_index=0) == 1
    assert select_checkpoint_frame(episode, checkpoint_frame=1) == 1


def test_select_checkpoint_frame_rejects_out_of_range_frame(tmp_path):
    episode = _write_episode(tmp_path / "teleop_data")

    with pytest.raises(IndexError, match="out of range"):
        select_checkpoint_frame(episode, checkpoint_frame=99)


def test_select_checkpoint_frame_falls_back_to_zero_for_old_episode_without_checkpoints(tmp_path):
    episode = _write_episode(tmp_path / "teleop_data")
    with h5py.File(episode, "a") as h5:
        del h5["checkpoints"]

    assert select_checkpoint_frame(episode) == 0


def test_replay_frame_range_defaults_to_checkpoint_through_episode_end():
    assert list(replay_frame_range(frame_count=5, checkpoint_frame=2, start_frame=None, stop_frame=None, max_steps=None)) == [2, 3, 4]
    assert list(replay_frame_range(frame_count=5, checkpoint_frame=2, start_frame=None, stop_frame=None, max_steps=2)) == [2, 3]
