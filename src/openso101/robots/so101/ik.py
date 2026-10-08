# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import math

import torch


def differential_ik(jacobian, delta, joint_position, joint_limits, *, dt, damping=0.05, max_speed=1.0):
    if any(not math.isfinite(value) or value <= 0 for value in (dt, damping, max_speed)):
        raise ValueError("dt、damping 和 max_speed 必须为正数有限数值")
    if jacobian.shape[-2] != 4 or delta.shape[-1] != 4:
        raise ValueError("IK 输入必须包含 xyz 和 yaw 四个分量")
    if jacobian.shape[:-2] != joint_position.shape[:-1] or jacobian.shape[-1] != joint_position.shape[-1]:
        raise ValueError("Jacobian 与关节位置形状不一致")
    if delta.shape[:-1] != joint_position.shape[:-1] or joint_limits.shape != (*joint_position.shape, 2):
        raise ValueError("目标增量或关节限位形状不一致")
    for value in (jacobian, delta, joint_position, joint_limits):
        if not torch.isfinite(value).all():
            raise ValueError("IK 输入必须为有限数值")
    lower, upper = joint_limits.unbind(-1)
    if (lower >= upper).any():
        raise ValueError("关节限位下限必须小于上限")
    if (joint_position < lower - 0.01).any() or (joint_position > upper + 0.01).any():
        raise ValueError(f"关节位置超出限位：position={joint_position.tolist()}, limits={joint_limits.tolist()}")
    transpose = jacobian.transpose(-1, -2)
    regularized = jacobian @ transpose + damping ** 2 * torch.eye(4, device=jacobian.device, dtype=jacobian.dtype)
    increment = (transpose @ torch.linalg.solve(regularized, delta.unsqueeze(-1))).squeeze(-1)
    increment = increment.clamp(-max_speed * dt, max_speed * dt)
    return torch.maximum(lower, torch.minimum(upper, joint_position + increment))
