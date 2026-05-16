# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Runtime LeRobot wrappers for OpenSO-101 teleoperation.

The :class:`LeRobotSO101Leader` can read the physical leader synchronously
(blocking the sim thread on each Feetech bus read) or asynchronously via a
daemon thread that polls the leader in the background and caches the latest
reading. Async mode restores the latency feature that was lost during the
``safe_sim2real`` → ``openso101`` port and mirrors the conflate-style
worker that ``leisaac/devices/lerobot/so101_leader_remote.py`` runs over
ZMQ — but for the local single-machine case a plain
:class:`threading.Thread` is enough since shared memory replaces ZMQ.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Mapping

from .so101_mapping import lerobot_action_to_ordered_targets, transform_ordered_targets


class LeRobotDependencyError(RuntimeError):
    """Raised when LeRobot is needed but is not installed in the active env."""


def _make_so101_leader(port: str, robot_id: str) -> Any:
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig
    except ImportError as exc:
        raise LeRobotDependencyError(
            "LeRobot SO101 support is not installed. Install LeRobot in this "
            "environment before running OpenSO-101 teleop."
        ) from exc

    return make_robot_from_config(SO101LeaderConfig(port=port, id=robot_id))


class LeRobotSO101Leader:
    """SO101 leader-arm reader with OpenSO-101 target conversion.

    Parameters
    ----------
    port, robot_id:
        Forwarded to ``SO101LeaderConfig``.
    robot:
        Pre-built leader handle for tests / dependency injection.
    inverted_joints, joint_offsets_rad:
        Per-joint calibration applied after leader→sim mapping.
    async_read:
        When ``True``, a daemon thread polls the leader continuously and
        :meth:`read_action` returns the cached latest reading instead of
        blocking on the serial bus. Off by default so sim and leader stay
        on the same thread for the easiest debugging story; recommended on
        for live teleop where sub-millisecond leader latency matters.
    """

    def __init__(
        self,
        port: str,
        robot_id: str,
        robot: Any | None = None,
        *,
        inverted_joints=(),
        joint_offsets_rad: Mapping[str, float] | None = None,
        async_read: bool = False,
    ):
        self.port = port
        self.robot_id = robot_id
        self._robot = robot
        self.inverted_joints = tuple(inverted_joints)
        self.joint_offsets_rad = dict(joint_offsets_rad or {})

        self._async_read = bool(async_read)
        self._async_thread: threading.Thread | None = None
        self._async_stop = threading.Event()
        self._async_lock = threading.Lock()
        self._cached_action: Mapping[str, float] | None = None
        self._async_error: BaseException | None = None
        self._async_reads = 0

    @property
    def robot(self) -> Any:
        if self._robot is None:
            self._robot = _make_so101_leader(self.port, self.robot_id)
        return self._robot

    @property
    def async_read(self) -> bool:
        return self._async_read

    @property
    def async_read_count(self) -> int:
        """How many leader reads the background thread has completed.

        Exposed for diagnostics — the teleop profile path prints this so
        users can verify the worker is actually running.
        """
        with self._async_lock:
            return self._async_reads

    def connect(self) -> None:
        self.robot.connect()
        if self._async_read:
            self._start_async_worker()

    def close(self) -> None:
        """Stop the worker thread (if running) and let the OS reclaim it."""
        self._stop_async_worker()

    # --- Async worker ----------------------------------------------------

    def _start_async_worker(self) -> None:
        if self._async_thread is not None:
            return
        # Prime the cache with one synchronous read so the first call to
        # :meth:`read_action` after ``connect()`` doesn't return None.
        self._cached_action = self.robot.get_action()
        self._async_stop.clear()
        self._async_thread = threading.Thread(
            target=self._async_loop,
            name=f"so101-leader-{self.robot_id}",
            daemon=True,
        )
        self._async_thread.start()

    def _stop_async_worker(self) -> None:
        thread = self._async_thread
        if thread is None:
            return
        self._async_stop.set()
        thread.join(timeout=1.0)
        self._async_thread = None

    def _async_loop(self) -> None:
        """Tight poll loop: read leader, cache it, repeat until stop.

        Errors from the bus are stored on ``self._async_error`` so the main
        thread can surface them on the next read instead of dying silently.
        We sleep 1 ms between iterations so other threads aren't starved
        on single-core machines; the Feetech bus read itself is ~5–20 ms
        so this adds negligible overhead.
        """
        while not self._async_stop.is_set():
            try:
                action = self.robot.get_action()
            except BaseException as exc:  # noqa: BLE001 — propagate to main
                with self._async_lock:
                    self._async_error = exc
                return
            with self._async_lock:
                self._cached_action = action
                self._async_reads += 1
            # Yield briefly. Leader hardware update rate is the upper
            # bound; the cooperative sleep keeps the loop from pegging a
            # core during long sessions.
            time.sleep(0.001)

    # --- Public read API -------------------------------------------------

    def read_action(self) -> Mapping[str, float]:
        if self._async_read:
            with self._async_lock:
                if self._async_error is not None:
                    raise self._async_error
                if self._cached_action is None:
                    # Worker hasn't produced a frame yet — fall back to a
                    # one-shot synchronous read so callers don't see None.
                    return self.robot.get_action()
                # Return a shallow copy so callers can mutate freely
                # without racing the writer thread.
                return copy.copy(self._cached_action)
        return self.robot.get_action()

    def read_ordered_targets(self) -> tuple[Mapping[str, float], tuple[float, ...]]:
        action = self.read_action()
        targets = lerobot_action_to_ordered_targets(action)
        targets = transform_ordered_targets(
            targets,
            inverted_joints=self.inverted_joints,
            offsets_rad=self.joint_offsets_rad,
        )
        return action, targets

    def read_target_tensor(self, device: str):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to create Isaac action tensors.") from exc

        action, targets = self.read_ordered_targets()
        return action, torch.tensor(targets, dtype=torch.float32, device=device)
