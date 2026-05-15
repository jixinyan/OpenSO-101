import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_hdf5_teleop_recorder.py for the legacy assertions.",
    allow_module_level=True,
)

import h5py
import numpy as np

from openso101.teleop.hdf5_recorder import SafeSim2RealHDF5TeleopRecorder, validate_hdf5_dataset
from openso101.robots import SO101_SIM_JOINT_NAMES


CAMERAS = {
    "wrist_camera": {"height": 2, "width": 3},
    "overhead_camera": {"height": 2, "width": 3},
}


def test_hdf5_recorder_writes_lerobot_compatible_episode_layout(tmp_path):
    recorder = SafeSim2RealHDF5TeleopRecorder(
        root=tmp_path / "teleop_data" / "safe_sim2real_pickplace_teleop",
        task_name="Pick up the green cube and place it at the goal",
        cameras=CAMERAS,
        fps=30,
    )
    recorder.start_episode()
    recorder.add_frame(
        action=np.arange(6, dtype=np.float32),
        qpos=np.arange(6, dtype=np.float32) + 0.1,
        qvel=np.arange(6, dtype=np.float32) + 0.2,
        camera_buffers={
            "wrist_camera": np.zeros((2, 3, 3), dtype=np.uint8),
            "overhead_camera": np.ones((2, 3, 3), dtype=np.uint8),
        },
        timestamp=1.25,
    )
    episode_path = recorder.save_episode(success=True)

    assert episode_path == tmp_path / "teleop_data" / "safe_sim2real_pickplace_teleop" / "episodes" / "episode_000000.hdf5"
    with h5py.File(episode_path, "r") as h5:
        assert h5["action"].shape == (1, 6)
        assert h5["observations/qpos"].shape == (1, 6)
        assert h5["observations/qvel"].shape == (1, 6)
        assert h5["observations/images/wrist_camera"].shape == (1, 2, 3, 3)
        assert h5["observations/images/overhead_camera"].shape == (1, 2, 3, 3)
        assert h5["timestamps"][0] == pytest.approx(1.25)
        assert h5.attrs["task"] == "Pick up the green cube and place it at the goal"
        assert h5.attrs["fps"] == 30
        assert bool(h5.attrs["success"]) is True
        assert list(h5.attrs["camera_names"]) == ["wrist_camera", "overhead_camera"]
        assert h5["checkpoints/frame_index"].shape == (0,)


def test_hdf5_recorder_uses_next_episode_index(tmp_path):
    root = tmp_path / "teleop_data" / "safe_sim2real_pickplace_teleop"
    episodes = root / "episodes"
    episodes.mkdir(parents=True)
    (episodes / "episode_000000.hdf5").write_bytes(b"existing")

    recorder = SafeSim2RealHDF5TeleopRecorder(root=root, task_name="task", cameras=CAMERAS, fps=30)

    assert recorder.next_episode_path() == episodes / "episode_000001.hdf5"


def test_validate_hdf5_dataset_requires_complete_episode_layout(tmp_path):
    root = tmp_path / "teleop_data"
    episode = root / "episodes" / "episode_000000.hdf5"
    episode.parent.mkdir(parents=True)
    with h5py.File(episode, "w") as h5:
        h5.create_dataset("action", data=np.zeros((1, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="observations/qpos"):
        validate_hdf5_dataset(root)


def test_hdf5_recorder_checkpoint_restore_truncates_current_episode(tmp_path):
    recorder = SafeSim2RealHDF5TeleopRecorder(
        root=tmp_path / "teleop_data",
        task_name="task",
        cameras=CAMERAS,
        fps=30,
    )
    recorder.start_episode()
    recorder.add_frame(
        action=np.zeros(6, dtype=np.float32),
        qpos=np.zeros(6, dtype=np.float32),
        qvel=np.zeros(6, dtype=np.float32),
        camera_buffers={
            "wrist_camera": np.zeros((2, 3, 3), dtype=np.uint8),
            "overhead_camera": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        timestamp=0.0,
    )
    checkpoint = recorder.create_checkpoint()
    recorder.add_frame(
        action=np.ones(6, dtype=np.float32),
        qpos=np.ones(6, dtype=np.float32),
        qvel=np.ones(6, dtype=np.float32),
        camera_buffers={
            "wrist_camera": np.ones((2, 3, 3), dtype=np.uint8),
            "overhead_camera": np.ones((2, 3, 3), dtype=np.uint8),
        },
        timestamp=1.0,
    )

    recorder.restore_checkpoint(checkpoint)
    episode_path = recorder.save_episode()

    with h5py.File(episode_path, "r") as h5:
        assert h5["action"].shape == (1, 6)
        assert np.all(h5["action"][0] == 0.0)
        assert h5["checkpoints/frame_index"][:].tolist() == [0]


def test_hdf5_recorder_writes_optional_sim_state_for_checkpoint_replay(tmp_path):
    recorder = SafeSim2RealHDF5TeleopRecorder(
        root=tmp_path / "teleop_data",
        task_name="task",
        cameras=CAMERAS,
        fps=30,
    )
    recorder.start_episode()
    recorder.add_frame(
        action=np.zeros(6, dtype=np.float32),
        qpos=np.zeros(6, dtype=np.float32),
        qvel=np.zeros(6, dtype=np.float32),
        camera_buffers={
            "wrist_camera": np.zeros((2, 3, 3), dtype=np.uint8),
            "overhead_camera": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        timestamp=0.0,
        sim_state={
            "object_root_state": np.arange(13, dtype=np.float32),
            "command_stage": np.asarray(2, dtype=np.int64),
            "command_goal_pos_b": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "command_goal_pos_w": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
            "command_cube_spawn_xy_b": np.asarray([0.7, 0.8], dtype=np.float32),
        },
    )

    episode_path = recorder.save_episode()

    with h5py.File(episode_path, "r") as h5:
        assert h5["sim/object_root_state"].shape == (1, 13)
        assert h5["sim/command_stage"].shape == (1,)
        assert h5["sim/command_goal_pos_b"].shape == (1, 3)
        assert h5["sim/command_goal_pos_w"].shape == (1, 3)
        assert h5["sim/command_cube_spawn_xy_b"].shape == (1, 2)


def test_hdf5_recorder_writes_actual_sim_joint_names(tmp_path):
    recorder = SafeSim2RealHDF5TeleopRecorder(
        root=tmp_path / "teleop_data",
        task_name="task",
        cameras=CAMERAS,
        fps=30,
        sim_joint_names=SO101_SIM_JOINT_NAMES,
    )
    recorder.start_episode()
    recorder.add_frame(
        action=np.zeros(6, dtype=np.float32),
        qpos=np.zeros(6, dtype=np.float32),
        qvel=np.zeros(6, dtype=np.float32),
        camera_buffers={
            "wrist_camera": np.zeros((2, 3, 3), dtype=np.uint8),
            "overhead_camera": np.zeros((2, 3, 3), dtype=np.uint8),
        },
        timestamp=0.0,
    )

    episode_path = recorder.save_episode()

    with h5py.File(episode_path, "r") as h5:
        assert list(h5.attrs["sim_joint_names"]) == list(SO101_SIM_JOINT_NAMES)
