import pytest

import math

from openso101.teleop.so101_mapping import SO101_TELEOP_TARGET_LIMITS_DEG


class FakeLeaderRobot:
    def __init__(self):
        self.connected = False

    def connect(self):
        self.connected = True

    def get_action(self):
        return {
            "shoulder_pan.pos": 100.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,
        }


def test_leader_wrapper_connects_injected_robot_and_maps_targets():
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader

    fake_robot = FakeLeaderRobot()
    leader = LeRobotSO101Leader(port="/dev/null", robot_id="test_leader", robot=fake_robot)

    leader.connect()
    action, targets = leader.read_ordered_targets()

    assert fake_robot.connected is True
    assert action["shoulder_pan.pos"] == 100.0
    assert targets[0] == pytest.approx(math.radians(SO101_TELEOP_TARGET_LIMITS_DEG["shoulder_pan"].upper))


# ---------------------------------------------------------------------------
# Async leader read (daemon thread + cached latest)
# ---------------------------------------------------------------------------


class CountingLeaderRobot:
    """Counts get_action() calls so we can verify which thread is reading."""

    def __init__(self):
        import threading
        self.connected = False
        self.call_count = 0
        self._lock = threading.Lock()
        # Allow tests to mutate the position the robot reports without race.
        self._pos = 0.0

    def connect(self):
        self.connected = True

    def set_pos(self, value: float) -> None:
        with self._lock:
            self._pos = float(value)

    def get_action(self):
        with self._lock:
            self.call_count += 1
            pos = self._pos
        return {
            "shoulder_pan.pos": pos,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }


def test_sync_read_calls_robot_get_action_each_time():
    """Default (async_read=False): each read_action() hits the bus."""
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader

    robot = CountingLeaderRobot()
    leader = LeRobotSO101Leader(port="/dev/null", robot_id="t", robot=robot)
    leader.connect()

    for _ in range(3):
        leader.read_action()

    assert leader.async_read is False
    assert robot.call_count == 3


def test_async_read_spawns_worker_and_caches_latest():
    """async_read=True: the daemon thread polls; main reads cached value."""
    import time
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader

    robot = CountingLeaderRobot()
    leader = LeRobotSO101Leader(
        port="/dev/null", robot_id="t", robot=robot, async_read=True,
    )
    try:
        leader.connect()
        # The worker primes the cache via one sync read; give the thread a
        # moment to perform a few additional reads.
        time.sleep(0.05)
        snapshot_one = leader.read_action()
        reads_after_first = leader.async_read_count

        # Change the underlying robot state; the worker should pick it up
        # within the next iteration (we poll at ~1 ms cadence).
        robot.set_pos(42.0)
        time.sleep(0.05)
        snapshot_two = leader.read_action()

        assert leader.async_read is True
        assert snapshot_one["shoulder_pan.pos"] == pytest.approx(0.0)
        assert snapshot_two["shoulder_pan.pos"] == pytest.approx(42.0)
        # Worker should have read many times during 50 ms of polling.
        assert reads_after_first >= 2
    finally:
        leader.close()


def test_async_read_close_joins_worker_thread():
    """close() must stop the daemon thread."""
    import time
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader

    robot = CountingLeaderRobot()
    leader = LeRobotSO101Leader(
        port="/dev/null", robot_id="t", robot=robot, async_read=True,
    )
    leader.connect()
    time.sleep(0.02)
    leader.close()

    # After close, the worker thread is gone and the count is frozen.
    frozen_count = leader.async_read_count
    time.sleep(0.05)
    assert leader.async_read_count == frozen_count


def test_async_read_surfaces_worker_errors_on_next_read():
    """If the bus read throws inside the worker, the next main-thread
    read_action() must re-raise so we don't silently lose data."""
    import time
    from openso101.teleop.lerobot_interface import LeRobotSO101Leader

    class FlakyRobot(CountingLeaderRobot):
        def __init__(self):
            super().__init__()
            self._first_call = True

        def get_action(self):
            if self._first_call:
                self._first_call = False
                return super().get_action()
            raise RuntimeError("simulated bus failure")

    robot = FlakyRobot()
    leader = LeRobotSO101Leader(
        port="/dev/null", robot_id="t", robot=robot, async_read=True,
    )
    leader.connect()
    # Give the worker a moment to hit the second get_action() and store the error.
    time.sleep(0.05)
    try:
        with pytest.raises(RuntimeError, match="simulated bus failure"):
            leader.read_action()
    finally:
        leader.close()
