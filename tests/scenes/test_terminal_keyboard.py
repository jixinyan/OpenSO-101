import os
import termios
import time

import pytest

from openso101.teleop.devices.terminal import TerminalKeyboard
from openso101.cli.il import _TeleopKeyboard
from openso101.cli.main import build_parser


def test_terminal_keys_expire_and_restore_terminal():
    master, slave = os.openpty()
    with os.fdopen(slave, "r") as stream:
        previous = termios.tcgetattr(stream.fileno())
        reader = TerminalKeyboard(stream, release_seconds=0.02)
        try:
            reader.connect()
            os.write(master, b"\x1b[A\x1b[5~a gcq")
            pressed, names = reader.poll()
            assert pressed == {"UP", "PAGE_UP", "A"}
            assert names == ("UP", "PAGE_UP", "A", "SPACE", "G", "C", "Q")
            requests = _TeleopKeyboard(subscribe=False)
            for name in names:
                requests.request_key(name)
            assert requests.take_checkpoint and requests.quit_discard
            assert not requests.mark_success
            time.sleep(0.03)
            assert reader.poll() == (set(), ())
        finally:
            reader.disconnect()
            try:
                assert termios.tcgetattr(stream.fileno()) == previous
            finally:
                os.close(master)


def test_fragmented_terminal_sequence_uses_library_decoder():
    master, slave = os.openpty()
    with os.fdopen(slave, "r") as stream:
        reader = TerminalKeyboard(stream)
        try:
            reader.connect()
            os.write(master, b"\x1b[")
            assert reader.poll() == (set(), ())
            os.write(master, b"D")
            assert reader.poll() == ({"LEFT"}, ("LEFT",))
        finally:
            reader.disconnect()
            os.close(master)


def test_terminal_rejects_noninteractive_input(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("up")
    with path.open() as stream, pytest.raises(ValueError, match="交互终端"):
        TerminalKeyboard(stream)


def test_record_parser_exposes_terminal_keyboard():
    args = build_parser().parse_args([
        "il", "record", "--task", "OpenSO101-Lift-v0", "--teleop-device", "keyboard",
        "--keyboard-input", "terminal", "--headless", "--no-camera-viewports",
    ])
    assert args.headless and args.keyboard_input == "terminal"
