import pytest

pytest.skip(
    "ported from safe_sim2real under OpenSO-101 skeleton mode; "
    "awaits implementation port. See "
    "/data/safe_sim2real/tests/test_teleop_success.py for the legacy assertions.",
    allow_module_level=True,
)

import numpy as np

from openso101.scripts.lerobot.teleop_agent import _prompt_save_successful_episode, _success_prompt_response


def test_success_prompt_response_accepts_yes_answers():
    assert _success_prompt_response("y") is True
    assert _success_prompt_response("Yes") is True


def test_success_prompt_response_rejects_default_and_no_answers():
    assert _success_prompt_response("") is False
    assert _success_prompt_response("n") is False


class FakeRecorder:
    def __init__(self):
        self.recording = True
        self.calls = []

    def save_episode(self, success=False):
        self.calls.append(("save", success))
        self.recording = False

    def cancel_episode(self):
        self.calls.append("cancel")
        self.recording = False


def test_prompt_save_successful_episode_saves_when_user_accepts():
    recorder = FakeRecorder()

    _prompt_save_successful_episode(recorder, input_fn=lambda _: "y")

    assert recorder.calls == [("save", True)]


def test_prompt_save_successful_episode_cancels_when_user_declines():
    recorder = FakeRecorder()

    _prompt_save_successful_episode(recorder, input_fn=lambda _: "n")

    assert recorder.calls == ["cancel"]
