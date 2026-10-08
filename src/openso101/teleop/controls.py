import math
from dataclasses import dataclass
from typing import Any


def _clone_value(value):
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _coerce_target_like(target, reference):
    import numpy as np
    import torch

    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(target, device=reference.device, dtype=reference.dtype)
    if isinstance(reference, np.ndarray):
        values = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target
        return np.asarray(values, dtype=reference.dtype)
    values = torch.as_tensor(target, dtype=torch.float64).tolist()
    return tuple(values) if isinstance(reference, tuple) else values


def _target_max_abs_error(a, b) -> float:
    import torch

    left = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=torch.float64)
    right = torch.as_tensor(b, device=left.device, dtype=left.dtype)
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("关节目标的形状需要一致且包含数值")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("关节目标需要有限数值")
    return float((left - right).abs().max())


def _nonnegative_limit(name, value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} 必须为非负有限数值")
    return result


@dataclass
class _ResumeHoldState:
    targets: Any
    holding: bool
    released: bool
    error: float | None


class _TeleopResumeHold:
    def __init__(self, release_threshold: float, *, device: str = "leader"):
        if device not in ("leader", "keyboard"):
            raise ValueError("遥操设备需要 leader 或 keyboard")
        self.device = device
        self.default_threshold = _nonnegative_limit("release_threshold", release_threshold)
        self.release_threshold = self.default_threshold
        self._target = None
        self._context = "checkpoint"

    @property
    def active(self) -> bool:
        return self._target is not None

    def activate(self, target, *, context: str = "checkpoint", release_threshold: float | None = None) -> None:
        if context not in ("startup", "checkpoint"):
            raise ValueError("遥操保持状态需要 startup 或 checkpoint")
        _target_max_abs_error(target, target)
        threshold = (self.default_threshold if release_threshold is None else
                     _nonnegative_limit("release_threshold", release_threshold))
        self._target = _clone_value(target)
        self._context = context
        self.release_threshold = threshold
        if self.device == "keyboard":
            print("[INFO]: 键盘控制保持在当前记录的姿态。")
        elif context == "startup":
            print("[INFO]: 机器人保持在初始姿态。将 leader arm 移动到该姿态后开始遥操。")
        else:
            print("[INFO]: 机器人保持在 checkpoint 姿态。将 leader arm 移动到该姿态后恢复遥操。")

    def apply(self, leader_targets) -> _ResumeHoldState:
        _target_max_abs_error(leader_targets, leader_targets)
        if self._target is None:
            return _ResumeHoldState(leader_targets, False, False, None)
        hold_target = _coerce_target_like(self._target, leader_targets)
        error = _target_max_abs_error(leader_targets, hold_target)
        if error <= self.release_threshold:
            self._target = None
            label = "键盘控制" if self.device == "keyboard" else "leader arm 控制"
            print(f"[INFO]: {label}已恢复，最大关节目标误差为 {error:.4f} rad。")
            return _ResumeHoldState(leader_targets, False, True, error)
        return _ResumeHoldState(hold_target, True, False, error)


class _TeleopTargetRateLimiter:
    def __init__(self, max_delta: float):
        self.max_delta = _nonnegative_limit("max_delta", max_delta)
        self._previous = None

    @property
    def enabled(self) -> bool:
        return self.max_delta > 0

    def reset(self, target=None) -> None:
        if target is not None:
            _target_max_abs_error(target, target)
        self._previous = _clone_value(target) if target is not None else None

    def apply(self, target):
        import torch

        _target_max_abs_error(target, target)
        if not self.enabled:
            return target
        if self._previous is None:
            self._previous = _clone_value(target)
            return target
        previous = _coerce_target_like(self._previous, target)
        _target_max_abs_error(target, previous)
        values = target if isinstance(target, torch.Tensor) else torch.as_tensor(target, dtype=torch.float64)
        reference = torch.as_tensor(previous, device=values.device, dtype=values.dtype)
        limited = _coerce_target_like(reference + (values - reference).clamp(-self.max_delta, self.max_delta), target)
        self._previous = _clone_value(limited)
        return limited
