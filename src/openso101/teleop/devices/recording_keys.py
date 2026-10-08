class _TeleopKeyboard:
    def __init__(self, *, subscribe=True):
        self.take_checkpoint = False
        self.restore_checkpoint = False
        self.mark_success = False
        self.quit_discard = False
        self._sub_keyboard = None
        if not subscribe:
            return
        import carb.input
        import omni.appwindow

        self._keyboard_event_type = carb.input.KeyboardEventType
        self._window = omni.appwindow.get_default_app_window()
        if self._window is None:
            raise RuntimeError("窗口键盘需要 Isaac 图形窗口")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    @property
    def checkpoint_recording(self) -> bool:
        return self.take_checkpoint

    @checkpoint_recording.setter
    def checkpoint_recording(self, value: bool) -> None:
        self.take_checkpoint = value

    @property
    def resume_recording(self) -> bool:
        return self.restore_checkpoint

    @resume_recording.setter
    def resume_recording(self, value: bool) -> None:
        self.restore_checkpoint = value

    @property
    def toggle_recording(self) -> bool:
        return self.mark_success

    @toggle_recording.setter
    def toggle_recording(self, value: bool) -> None:
        self.mark_success = value

    @property
    def quit_without_saving(self) -> bool:
        return self.quit_discard

    @quit_without_saving.setter
    def quit_without_saving(self, value: bool) -> None:
        self.quit_discard = value

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != self._keyboard_event_type.KEY_PRESS:
            return False
        return self.request_key(event.input.name.upper())

    def request_key(self, key_name):
        fields = {"C": "take_checkpoint", "R": "restore_checkpoint", "Q": "quit_discard", "S": "mark_success"}
        if key_name not in fields:
            return False
        setattr(self, fields[key_name], True)
        print(f"[INFO]: 已接收录制按键: {key_name}")
        return True

    def cleanup(self) -> None:
        if self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
