import torch


def camera_to_policy(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[0] < 1 or rgb.shape[1] < 1 or rgb.shape[2] < 1 or rgb.shape[3] not in (3, 4):
        raise ValueError("相机需要 [N, H, W, 3/4] RGB 数据")
    rgb = rgb[..., :3]
    if rgb.dtype == torch.uint8:
        pixels = rgb.to(torch.float32) / 255.0
    elif rgb.is_floating_point():
        if not torch.isfinite(rgb).all() or bool((rgb < 0).any()) or bool((rgb > 1).any()):
            raise ValueError("浮点相机 RGB 需要位于零与一之间")
        pixels = rgb.to(torch.float32)
    else:
        raise ValueError("相机 RGB dtype 需要为 uint8 或浮点数")
    return pixels.permute(0, 3, 1, 2).contiguous()


def _build_il_policy_observation_batched(scene) -> dict:
    from openso101.teleop.recorder.lerobot import REQUIRED_CAMERA_NAMES, get_scene_entity
    from openso101.teleop.so101_mapping import batched_action_to_motor_units, get_sim_joint_names

    robot = scene["robot"]
    joint_names = list(robot.joint_names)
    indices = [joint_names.index(name) for name in get_sim_joint_names()]
    positions = robot.data.joint_pos[:, indices].to(torch.float32)
    if positions.ndim != 2 or positions.shape[1] != 6 or not torch.isfinite(positions).all():
        raise ValueError("IL observation 需要有限的 [N, 6] 关节位置")
    observation = {"observation.state": batched_action_to_motor_units(positions)}
    for name in REQUIRED_CAMERA_NAMES:
        pixels = camera_to_policy(get_scene_entity(scene, name).data.output["rgb"])
        if pixels.shape[0] != positions.shape[0]:
            raise ValueError(f"相机与机器人环境数量不一致: {name}")
        observation[f"observation.images.{name}"] = pixels
    return observation


def _build_il_policy_observation(unwrapped_env, scene) -> dict:
    return {name: value[:1] for name, value in _build_il_policy_observation_batched(scene).items()}
