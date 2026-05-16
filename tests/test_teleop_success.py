import pytest

import numpy as np

from openso101.cli.il import _handle_successful_episode, _success_prompt_response


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


def test_interactive_prompt_is_default_on_success():
    """Default behavior: prompt the user with [y/N]. Save only on yes."""
    recorder = FakeRecorder()

    _handle_successful_episode(recorder, input_fn=lambda _: "y")

    assert recorder.calls == [("save", True)]


def test_interactive_prompt_cancels_on_no_answer():
    recorder = FakeRecorder()

    _handle_successful_episode(recorder, input_fn=lambda _: "n")

    assert recorder.calls == ["cancel"]


def test_auto_save_skips_prompt_when_confirm_is_false():
    """confirm=False (i.e. --auto-save) skips the prompt and saves."""
    recorder = FakeRecorder()
    sentinel = []

    def input_fn(_):
        sentinel.append("called")
        return "n"

    _handle_successful_episode(recorder, confirm=False, input_fn=input_fn)

    assert recorder.calls == [("save", True)]
    assert sentinel == [], "input_fn should not be called in auto-save mode"
