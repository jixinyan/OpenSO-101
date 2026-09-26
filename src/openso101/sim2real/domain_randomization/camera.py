# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaacsim.core.prims import XFormPrim


def randomize_camera_mounts(env, env_ids, position_range_m=0.003, rotation_range_rad=0.02):
    if position_range_m < 0 or rotation_range_rad < 0:
        raise ValueError("相机安装误差范围不能为负数")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "_camera_mounts"):
        env._camera_mounts = {}
        for name in ("wrist_camera", "overhead_camera"):
            view = XFormPrim(env.scene[name].cfg.prim_path, reset_xform_properties=False)
            view.initialize()
            position, rotation = view.get_local_poses()
            env._camera_mounts[name] = (view, position.clone(), rotation.clone())
    for view, position, rotation in env._camera_mounts.values():
        translation = (torch.rand((len(env_ids), 3), device=env.device) * 2 - 1) * position_range_m
        angles = (torch.rand((len(env_ids), 3), device=env.device) * 2 - 1) * rotation_range_rad
        delta = quat_from_euler_xyz(*angles.unbind(-1))
        view.set_local_poses(position[env_ids] + translation, quat_mul(rotation[env_ids], delta), env_ids)
