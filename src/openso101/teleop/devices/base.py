from dataclasses import dataclass
from typing import Protocol

import torch


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
