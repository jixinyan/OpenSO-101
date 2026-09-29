import math
import select
import sys
import time
from contextlib import ExitStack

from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys


class TerminalKeyboard:
    movement_keys = frozenset(("UP", "DOWN", "LEFT", "RIGHT", "PAGE_UP", "PAGE_DOWN", "A", "D"))
    key_names = {
        Keys.Up: "UP", Keys.Down: "DOWN", Keys.Left: "LEFT", Keys.Right: "RIGHT",
        Keys.PageUp: "PAGE_UP", Keys.PageDown: "PAGE_DOWN", Keys.ControlC: "Q",
        Keys.ControlD: "Q", " ": "SPACE",
    }

    def __init__(self, stream=None, *, release_seconds=0.15):
        if not math.isfinite(release_seconds) or release_seconds <= 0:
            raise ValueError("终端按键有效时间必须为有限正数")
        self.stream = sys.stdin if stream is None else stream
        if not self.stream.isatty():
            raise ValueError("终端键盘需要交互终端；SSH 使用 -tt")
        self.release_seconds = release_seconds
        self.expires = {}
        self.input = None
        self.contexts = ExitStack()

    def connect(self):
        if self.input is not None:
            raise RuntimeError("终端键盘已经连接")
        self.input = create_input(stdin=self.stream)
        self.contexts.enter_context(self.input.raw_mode())

    def poll(self):
        if self.input is None:
            raise RuntimeError("终端键盘尚未连接")
        if self.input.closed:
            raise EOFError("终端键盘输入已关闭")
        now = time.monotonic()
        names = []
        if select.select([self.input.fileno()], [], [], 0)[0]:
            for key in self.input.read_keys():
                name = self.key_names.get(key.key, key.data.upper())
                names.append(name)
                if name in self.movement_keys:
                    self.expires[name] = now + self.release_seconds
        self.expires = {name: expiry for name, expiry in self.expires.items() if expiry > now}
        return set(self.expires), tuple(names)

    def disconnect(self):
        self.contexts.close()
        if self.input is not None:
            self.input.close()
            self.input = None
        self.expires.clear()
