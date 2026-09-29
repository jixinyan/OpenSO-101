# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from openso101.robots.so101.constants import (
    SO101_ARM_JOINT_NAMES,
    SO101_SIM_JOINT_NAMES,
)
from openso101.robots.so101.ik import differential_ik
from openso101.teleop.so101_mapping import SO101_TELEOP_CONTROL_JOINT_NAMES


@dataclass(frozen=True)
class JointTargets:
    positions: torch.Tensor


@dataclass(frozen=True)
class CartesianDelta:
    delta: torch.Tensor
    gripper: float


class TeleopDevice(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_command(self) -> JointTargets | CartesianDelta: ...


class KeyboardDevice:
    async_read = False
    async_read_count = 0

    def __init__(self, env, *, translation_speed=0.06, rotation_speed=0.5, input_mode="window", on_key=None):
        if any(not math.isfinite(value) or value <= 0 for value in (translation_speed, rotation_speed)):
            raise ValueError("键盘移动速度必须为正数")
        if input_mode not in ("window", "terminal"):
            raise ValueError("keyboard input_mode 必须为 window 或 terminal")
        self.env = env
        self.robot = env.scene["robot"]
        self.translation_speed = translation_speed
        self.rotation_speed = rotation_speed
        self.pressed = set()
        self.gripper = 0.8
        self.arm_ids = [self.robot.joint_names.index(name) for name in SO101_ARM_JOINT_NAMES]
        self.control_ids = [self.robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
        self.body_id = self.robot.body_names.index("gripper")
        self.subscription = None
        self.input_mode = input_mode
        self.on_key = on_key
        self.terminal = None

    def connect(self):
        if self.input_mode == "terminal":
            from openso101.teleop.terminal import TerminalKeyboard

            self.terminal = TerminalKeyboard()
            self.terminal.connect()
            return
        import carb.input
        import omni.appwindow

        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_event)

    def _on_event(self, event):
        import carb.input

        name = event.input.name
        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            self.pressed.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(name)
        return True

    def get_command(self):
        if self.terminal is not None:
            self.pressed, names = self.terminal.poll()
            for name in names:
                if name == "SPACE":
                    self.gripper = 0.8
                elif name == "G":
                    self.gripper = 0.0
                if self.on_key is not None:
                    self.on_key(name)
        values = [
            float("UP" in self.pressed) - float("DOWN" in self.pressed),
            float("LEFT" in self.pressed) - float("RIGHT" in self.pressed),
            float("PAGE_UP" in self.pressed) - float("PAGE_DOWN" in self.pressed),
            float("A" in self.pressed) - float("D" in self.pressed),
        ]
        if "SPACE" in self.pressed:
            self.gripper = 0.8
        if "LEFT_SHIFT" in self.pressed or "RIGHT_SHIFT" in self.pressed or "G" in self.pressed:
            self.gripper = 0.0
        delta = torch.tensor(values, device=self.env.device)
        delta[:3] *= self.translation_speed * self.env.step_dt
        delta[3] *= self.rotation_speed * self.env.step_dt
        return CartesianDelta(delta=delta, gripper=self.gripper)

    def read_target_tensor(self, device):
        from isaaclab.utils.math import quat_apply

        command = self.get_command()
        body_index = self.body_id - int(self.robot.is_fixed_base)
        jacobian = self.robot.root_physx_view.get_jacobians()[:, body_index][:, :, self.arm_ids].clone()
        offset = torch.tensor((0.01, 0.0, -0.09), device=device).expand(self.env.num_envs, -1)
        offset_world = quat_apply(self.robot.data.body_quat_w[:, self.body_id], offset)
        angular = jacobian[:, 3:, :].transpose(1, 2)
        jacobian[:, :3, :] += torch.linalg.cross(angular, offset_world[:, None, :].expand_as(angular)).transpose(1, 2)
        joint_position = self.robot.data.joint_pos[:, self.arm_ids]
        target = differential_ik(
            jacobian[:, [0, 1, 2, 5], :], command.delta.expand(self.env.num_envs, -1),
            joint_position, self.robot.data.joint_pos_limits[:, self.arm_ids], dt=self.env.step_dt,
        )
        targets = torch.cat((target, torch.full((self.env.num_envs, 1), command.gripper, device=device)), dim=-1)
        raw = {f"{name}.pos": float(torch.rad2deg(targets[0, i])) for i, name in enumerate(SO101_TELEOP_CONTROL_JOINT_NAMES)}
        return raw, targets[0]

    def disconnect(self):
        if self.terminal is not None:
            self.terminal.disconnect()
            self.terminal = None
            self.pressed.clear()
        if self.subscription is not None:
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
            self.subscription = None

    def close(self):
        self.disconnect()
