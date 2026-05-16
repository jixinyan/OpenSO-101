import pytest

from types import SimpleNamespace

from openso101.cli.il import (
    _TeleopResumeHold,
    _TeleopTargetRateLimiter,
    _handle_recording_key_events,
)


class FakeRecorder:
    def __init__(self, recording=False):
        self.recording = recording
        self.calls = []

    def start_episode(self):
        self.calls.append("start")
        self.recording = True

    def save_episode(self, success=False):
        self.calls.append(("save", success))
        self.recording = False

    def cancel_episode(self):
        self.calls.append("cancel")
        self.recording = False

    def create_checkpoint(self):
        self.calls.append("checkpoint")
        return 7

    def restore_checkpoint(self, checkpoint):
        self.calls.append(("restore", checkpoint))
        self.recording = True


class FakeCheckpointStore:
    def __init__(self, checkpoint=None, hold_target=None):
        self.checkpoint = checkpoint
        self.hold_target = hold_target
        self.calls = []

    def capture(self, recorder):
        self.calls.append("capture")
        self.checkpoint = recorder.create_checkpoint()

    def restore(self, recorder):
        self.calls.append("restore")
        recorder.restore_checkpoint(self.checkpoint)
        return self.hold_target

    @property
    def has_checkpoint(self):
        return self.checkpoint is not None


def test_checkpoint_key_captures_active_episode_without_saving_or_quitting():
    keyboard = SimpleNamespace(checkpoint_recording=True, resume_recording=False, toggle_recording=False, quit_without_saving=False)
    recorder = FakeRecorder(recording=True)
    checkpoints = FakeCheckpointStore()

    should_quit = _handle_recording_key_events(keyboard, recorder, checkpoints)

    assert should_quit is False
    assert recorder.calls == ["checkpoint"]
    assert checkpoints.calls == ["capture"]
    assert keyboard.checkpoint_recording is False


def test_resume_key_restores_last_checkpoint_when_available():
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=True, toggle_recording=False, quit_without_saving=False)
    recorder = FakeRecorder(recording=True)
    checkpoints = FakeCheckpointStore(checkpoint=3)

    should_quit = _handle_recording_key_events(keyboard, recorder, checkpoints)

    assert should_quit is False
    assert recorder.calls == [("restore", 3)]
    assert checkpoints.calls == ["restore"]
    assert keyboard.resume_recording is False


def test_resume_key_does_not_activate_leader_sync_hold():
    """Per the new semantics, R is a HARD restore: snap to checkpoint
    immediately and let the leader take over next frame. The resume_hold
    machinery is left untouched so unrelated callers (e.g. startup
    sync) keep working, but the R key path itself must not activate it."""
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=True, toggle_recording=False, quit_without_saving=False)
    recorder = FakeRecorder(recording=True)
    checkpoints = FakeCheckpointStore(checkpoint=3, hold_target=[0.1, 0.2])
    resume_hold = _TeleopResumeHold(release_threshold=0.05)

    should_quit = _handle_recording_key_events(keyboard, recorder, checkpoints, resume_hold)

    assert should_quit is False
    assert resume_hold.active is False
    assert checkpoints.calls == ["restore"]


def test_resume_hold_releases_only_after_leader_returns_near_checkpoint():
    resume_hold = _TeleopResumeHold(release_threshold=0.05)
    resume_hold.activate([0.1, 0.2])

    far_state = resume_hold.apply([0.1, 0.4])
    assert far_state.holding is True
    assert far_state.targets == [0.1, 0.2]
    assert resume_hold.active is True

    near_state = resume_hold.apply([0.12, 0.22])
    assert near_state.released is True
    assert near_state.targets == [0.12, 0.22]
    assert resume_hold.active is False


def test_target_rate_limiter_limits_large_per_step_jumps():
    limiter = _TeleopTargetRateLimiter(max_delta=0.1)

    assert limiter.apply([0.0, 0.0]) == [0.0, 0.0]
    assert limiter.apply([1.0, -1.0]) == [0.1, -0.1]
    assert limiter.apply([0.15, -0.25]) == [0.15, -0.2]


def test_target_rate_limiter_can_be_disabled():
    limiter = _TeleopTargetRateLimiter(max_delta=0.0)

    assert limiter.apply([1.0, -1.0]) == [1.0, -1.0]


def test_resume_key_without_checkpoint_warns_does_not_start_episode():
    """Per the new semantics, R is restore-to-checkpoint only — it no longer
    auto-starts a fresh episode when none exists. Recording always begins at
    `il record` launch, so there's no legitimate "paused, no checkpoint"
    state to recover from with R."""
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=True, toggle_recording=False, quit_without_saving=False)
    recorder = FakeRecorder(recording=False)
    checkpoints = FakeCheckpointStore()

    should_quit = _handle_recording_key_events(keyboard, recorder, checkpoints)

    assert should_quit is False
    assert recorder.calls == [], "R with no checkpoint must NOT start a new episode"
    assert keyboard.resume_recording is False


def test_resume_key_without_checkpoint_does_not_clear_active_episode():
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=True, toggle_recording=False, quit_without_saving=False)
    recorder = FakeRecorder(recording=True)
    checkpoints = FakeCheckpointStore()

    should_quit = _handle_recording_key_events(keyboard, recorder, checkpoints)

    assert should_quit is False
    assert recorder.calls == []
    assert recorder.recording is True


def test_quit_key_cancels_active_episode_and_requests_exit():
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=False, toggle_recording=False, quit_without_saving=True)
    recorder = FakeRecorder(recording=True)

    should_quit = _handle_recording_key_events(keyboard, recorder, FakeCheckpointStore())

    assert should_quit is True
    assert recorder.calls == ["cancel"]
    assert keyboard.quit_without_saving is False


def test_s_key_marks_success_saves_and_requests_exit():
    """S = manual SUCCESS: save with success=True and exit the teleop loop."""
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=False, toggle_recording=True, quit_without_saving=False)
    recorder = FakeRecorder(recording=True)

    should_quit = _handle_recording_key_events(keyboard, recorder, FakeCheckpointStore())

    assert should_quit is True
    assert recorder.calls == [("save", True)]
    assert keyboard.toggle_recording is False


def test_s_key_with_no_active_recording_warns_but_does_not_quit():
    keyboard = SimpleNamespace(checkpoint_recording=False, resume_recording=False, toggle_recording=True, quit_without_saving=False)
    recorder = FakeRecorder(recording=False)

    should_quit = _handle_recording_key_events(keyboard, recorder, FakeCheckpointStore())

    # No recording in progress → nothing to save and no quit triggered.
    assert should_quit is False
    assert recorder.calls == []
    assert keyboard.toggle_recording is False
