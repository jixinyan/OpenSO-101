# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--cameras", action="store_true")
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.resets) <= 0:
        raise ValueError("环境数量、步骤数量和 reset 次数必须大于零")
    if args.output.exists():
        raise FileExistsError(args.output)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=args.cameras).app
    import gymnasium as gym
    import torch

    from .runtime import CustomSceneEnvCfg, register_custom_scene, scene_states
    from .usd import verify_compilation
    from .models import file_digest

    compilation = verify_compilation(args.scene)
    register_custom_scene()
    cfg = CustomSceneEnvCfg()
    cfg.configure_play(True)
    cfg.configure_scene(args.scene)
    cfg.configure_cameras(args.cameras)
    cfg.scene.num_envs = args.num_envs
    env = gym.make("OpenSO101-CustomScene-v0", cfg=cfg)
    images = {}
    camera_checks = {}
    terminations = {"terminated": 0, "truncated": 0}
    try:
        for reset in range(args.resets):
            observation, _ = env.reset(seed=cfg.seed + reset)
            if not torch.isfinite(observation["policy"]).all():
                raise ValueError(f"reset {reset} 产生无效观测")
            states = scene_states(env.unwrapped)
            if not torch.isfinite(states).all():
                raise ValueError(f"reset {reset} 产生无效实体状态")
            for index, entity in enumerate(cfg.scene_spec.entities):
                delta = states[:, index, :3] - torch.tensor(entity.pose.position, device=env.unwrapped.device)
                lower, upper = torch.tensor(entity.reset_translation_m, device=env.unwrapped.device)
                if (delta < lower - 1e-5).any() or (delta > upper + 1e-5).any():
                    raise ValueError(f"reset {reset} 超出配置范围：{entity.entity_id}")
        for step in range(args.steps):
            actions = torch.rand(env.action_space.shape, device=env.unwrapped.device) * 0.4 - 0.2
            observation, reward, terminated, truncated, _ = env.step(actions)
            if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
                raise ValueError(f"控制步骤 {step} 产生无效观测或奖励")
            robot = env.unwrapped.scene["robot"]
            if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(robot.data.joint_vel).all() or not torch.isfinite(scene_states(env.unwrapped)).all():
                raise ValueError(f"控制步骤 {step} 产生无效机器人或实体状态")
            terminations["terminated"] += int(terminated.sum())
            terminations["truncated"] += int(truncated.sum())
            if args.cameras:
                for name in ("wrist_camera", "overhead_camera"):
                    pixels = env.unwrapped.scene[name].data.output["rgb"][..., :3].float()
                    if pixels.ndim != 4 or pixels.shape[0] != args.num_envs or pixels.shape[-1] != 3:
                        raise ValueError(f"相机图像形状无效：{name}")
                    pixel_std = pixels.flatten(1).std(dim=1)
                    if not torch.isfinite(pixels).all() or (pixel_std <= 0).any():
                        raise ValueError(f"相机图像无效：{name}")
                    images[name] = list(pixels.shape)
                    entry = camera_checks.setdefault(name, {"checked_frames": 0, "minimum_pixel_std": float("inf")})
                    entry["checked_frames"] += args.num_envs
                    entry["minimum_pixel_std"] = min(entry["minimum_pixel_std"], float(pixel_std.min()))
        report = {
            "scene_sha256": compilation["scene_sha256"], "status": "runtime_verified",
            "num_envs": args.num_envs, "steps": args.steps, "resets": args.resets,
            "seed": cfg.seed, "cameras": images,
            "camera_checks": camera_checks, "episode_terminations": terminations,
            "task_success_verified": False, "dataset_verified": False,
            "control_dt": env.unwrapped.step_dt,
            "worker_sha256": file_digest(Path(__file__)),
            "runtime_source_sha256": file_digest(Path(__file__).with_name("runtime.py")),
            "compilation_manifest_sha256": file_digest(args.scene / "compilation.json"),
            "observation_shape": list(observation["policy"].shape),
            "pending_checks": ["dynamic_stability", "path_reachability", "contact_geometry", "camera_visibility", "task_completion", "successful_collection"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
    finally:
        env.close()
    app.close()


if __name__ == "__main__":
    main()
