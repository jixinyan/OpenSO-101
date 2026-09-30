# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from .config import CheckpointMeta, TrainCfg, digest


def build_environment(args, *, training: bool, scene: Path | None = None, student: bool = False):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import openso101.tasks  # noqa: F401

    if scene:
        from openso101.scenes.runtime import register_custom_scene

        register_custom_scene()
        if args.task != "OpenSO101-CustomScene-v0":
            raise ValueError("--scene 需要使用 OpenSO101-CustomScene-v0")
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs or 16)
    if getattr(args, "task_profile", "default") == "grasp_v2":
        from openso101.tasks.shared.grasp_profile import configure_grasp_profile

        configure_grasp_profile(cfg, args.task)
    cfg.configure_play(not training)
    cfg.scene.num_envs = args.num_envs or 16
    if scene:
        cfg.configure_scene(scene)
    cfg.seed = args.seed if args.seed is not None else 42
    cfg.configure_cameras(bool(getattr(args, "with_cameras", False)))
    if getattr(args, "visual_dr", False):
        cfg.configure_visual_dr(True)
    if student:
        from .vision_distillation import StudentObservationsCfg

        cfg.observations.student = StudentObservationsCfg()
    video = training and getattr(args, "video", False)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if video else None)
    if video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(args._video_dir), step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length, disable_logger=True,
        )
    return env


def train(args):
    if args.logger not in (None, "tensorboard"):
        raise ValueError("统一 backend 入口使用 --logger tensorboard；其他日志服务通过现有训练入口使用")
    config = TrainCfg.model_validate_json(Path(args.train_config).read_text()) if args.train_config else TrainCfg()
    overrides = {"backend": args.backend or config.backend, "algo": args.algo}
    if args.max_iterations is not None:
        overrides["iterations"] = args.max_iterations
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = TrainCfg.model_validate(config.model_dump() | overrides)
    config.batch_size(args.num_envs or 16)
    git_sha = args.source_revision or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    if len(git_sha) != 40 or any(char not in "0123456789abcdef" for char in git_sha):
        raise ValueError("source_revision 必须为完整的 Git SHA")
    args.seed = config.seed
    resume = Path(args.load_run).resolve() if args.load_run else None
    previous = CheckpointMeta.read(resume) if resume else None
    args.task_profile = getattr(args, "task_profile", None) or (previous.task_profile if previous else "default")
    if previous and previous.task_profile != args.task_profile:
        raise ValueError("继续训练需要保持 task_profile")
    if args.resume and resume is None:
        raise ValueError("继续训练需要 --load_run 指向完整训练目录")
    if args.checkpoint:
        raise ValueError("backend 训练通过 --load_run 加载 checkpoint.json 中的模型")
    if previous and (previous.task_id != args.task or previous.config.backend != config.backend or previous.config.algo != config.algo):
        raise ValueError("继续训练的任务、backend 或算法不匹配")
    output = Path(args.output) if args.output else Path("logs") / config.backend / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    args._video_dir = output / "videos"
    scene = Path(args.scene).resolve() if args.scene else None
    if previous and previous.scene_sha256:
        recorded = resume / "scene"
        if scene is None:
            scene = recorded
    scene_sha = None
    if scene:
        from openso101.scenes.usd import verify_compilation

        scene_sha = verify_compilation(scene)["scene_sha256"]
        shutil.copytree(scene, output / "scene")
        scene = output / "scene"
    if previous and previous.scene_sha256 != scene_sha:
        raise ValueError("继续训练的场景版本不匹配")
    (output / "train.json").write_text(config.model_dump_json(indent=2))
    package = Path(__file__).resolve().parents[1]
    with ZipFile(output / "source.zip", "w") as archive:
        for source in sorted(package.rglob("*.py")):
            archive.write(source, source.relative_to(package.parent))

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless, enable_cameras=args.with_cameras or args.video).app
    env = None
    try:
        from isaaclab.utils.io import dump_yaml

        from .backends import get_backend

        env = build_environment(args, training=True, scene=scene)
        dump_yaml(str(output / "environment.yaml"), env.unwrapped.cfg)
        checkpoint = get_backend(config.backend).train(env, config, output, resume)
        files = {checkpoint.name: digest(checkpoint), "train.json": digest(output / "train.json")}
        for name in ("backend.json", "normalization.pkl", "replay.pkl", "environment.yaml", "source.zip"):
            if (output / name).exists():
                files[name] = digest(output / name)
        if scene:
            for path in scene.rglob("*"):
                if path.is_file():
                    files[path.relative_to(output).as_posix()] = digest(path)
        CheckpointMeta(
            task_id=args.task, task_profile=args.task_profile, config=config, git_sha=git_sha, checkpoint=checkpoint.name,
            files=files, scene_sha256=scene_sha,
            completed_transitions=(previous.completed_transitions if previous else 0)
            + config.iterations * config.rollout_steps * env.unwrapped.num_envs,
        ).write(output)
        CheckpointMeta.read(output)
        print(json.dumps({"run": str(output.resolve()), "checkpoint": checkpoint.name, "status": "trained"}))
    finally:
        if env is not None:
            env.close()
    app.close()
    return 0


def evaluate(args, *, play=False):
    folder = Path(args.checkpoint).resolve()
    meta = CheckpointMeta.read(folder)
    args.task_profile = meta.task_profile
    if args.task != meta.task_id:
        raise ValueError("checkpoint 与请求任务不匹配")
    scene = folder / "scene" if meta.scene_sha256 else None
    args.seed = getattr(args, "seed", meta.config.seed)
    episodes_requested = getattr(args, "n_episodes", 10)
    if episodes_requested <= 0:
        raise ValueError("n_episodes 必须大于零")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless).app
    env = None
    try:
        import torch

        from .backends import get_backend

        env = build_environment(args, training=False, scene=scene)
        policy = get_backend(meta.config.backend).load(env, folder)
        observation, _ = env.reset()
        returns = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        lengths = torch.zeros_like(returns, dtype=torch.int64)
        latched = torch.zeros_like(returns, dtype=torch.bool)
        records = []
        quotas = torch.full_like(lengths, episodes_requested // env.unwrapped.num_envs)
        quotas[:episodes_requested % env.unwrapped.num_envs] += 1
        completed = torch.zeros_like(lengths)
        progress = {}
        if args.task in ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"):
            from openso101.tasks.shared.grasp import object_grasped_by_jaws
            from openso101.tasks.shared.rl_defaults import SO101_CONTROLLED_OBJECT_MIN_HEIGHT

            progress = {name: torch.zeros_like(latched) for name in ("reached", "grasped", "lifted", "held_above_table")}
            if args.task == "OpenSO101-PickPlace-v0":
                progress.update({name: torch.zeros_like(latched) for name in ("carry_stage", "place_stage")})
        success_terms = [name for name in env.unwrapped.termination_manager.active_terms if "success" in name]
        if not success_terms and not play:
            raise ValueError("任务缺少 success termination，无法计算成功率")
        with torch.inference_mode():
            while len(records) < episodes_requested and app.is_running():
                if progress:
                    scene = env.unwrapped.scene
                    object_position = scene["object"].data.root_pos_w
                    ee_position = scene["ee_frame"].data.target_pos_w[:, 0, :]
                    progress["reached"] |= torch.linalg.vector_norm(object_position - ee_position, dim=-1) < 0.08
                    grasped = object_grasped_by_jaws(env.unwrapped)
                    above_table = object_position[:, 2] - scene.env_origins[:, 2] > SO101_CONTROLLED_OBJECT_MIN_HEIGHT
                    progress["grasped"] |= grasped
                    progress["lifted"] |= above_table
                    progress["held_above_table"] |= grasped & above_table
                    if "carry_stage" in progress:
                        stage = env.unwrapped.command_manager.get_term("object_pose").stage
                        progress["carry_stage"] |= stage >= 1
                        progress["place_stage"] |= stage >= 2
                actions = policy(observation)
                observation, reward, terminated, truncated, _ = env.step(actions)
                returns += reward
                lengths += 1
                for name in success_terms:
                    latched |= env.unwrapped.termination_manager.get_term(name)
                for index in (terminated | truncated).nonzero().flatten().tolist():
                    if completed[index] < quotas[index]:
                        records.append({"success": bool(latched[index]), "return": float(returns[index]),
                                        "steps": int(lengths[index]), "env_index": index,
                                        **{name: bool(value[index]) for name, value in progress.items()}})
                        completed[index] += 1
                    returns[index] = 0
                    lengths[index] = 0
                    latched[index] = False
                    for value in progress.values():
                        value[index] = False
        if len(records) != episodes_requested:
            raise RuntimeError("仿真在评估完成之前终止")
        result = {"task": args.task, "task_profile": meta.task_profile, "backend": meta.config.backend, "seed": args.seed,
                  "episodes": records, "success_rate": sum(record["success"] for record in records) / len(records),
                  "checkpoint_sha256": digest(folder / meta.checkpoint), "training_git_sha": meta.git_sha,
                  "completed_transitions": meta.completed_transitions,
                  "evaluation_git_sha": subprocess.run(
                      ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                  ).stdout.strip(),
                  "torch_version": torch.__version__, "device": env.unwrapped.device,
                  "num_envs": env.unwrapped.num_envs, "episode_allocation": quotas.tolist(),
                  "progress_sampling": "before_control_step" if progress else None,
                  "progress_rates": {name: sum(record[name] for record in records) / len(records) for name in progress}}
        report = folder / f"evaluation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        report.write_text(json.dumps(result, indent=2))
        print(json.dumps({"report": str(report), "success_rate": result["success_rate"]}))
    finally:
        if env is not None:
            env.close()
    app.close()
    return 0


def distill(args):
    teacher = Path(args.teacher_run).resolve()
    meta = CheckpointMeta.read(teacher)
    if meta.config.backend != "rsl_rl" or meta.config.algo != "ppo":
        raise ValueError("视觉蒸馏需要 rsl_rl PPO teacher")
    if args.iterations <= 0 or args.rollout_steps <= 0:
        raise ValueError("iterations 和 rollout_steps 必须大于零")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.task = meta.task_id
    args.task_profile = meta.task_profile
    args.seed = meta.config.seed
    args.with_cameras = True
    scene = teacher / "scene" if meta.scene_sha256 else None
    if scene:
        shutil.copytree(scene, output / "scene")

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless, enable_cameras=True).app
    env = None
    try:
        from .vision_distillation import train_vision_student

        env = build_environment(args, training=True, scene=scene, student=True)
        train_vision_student(env, teacher, output, args.iterations, args.rollout_steps)
        print(json.dumps({"student": str(output), "status": "trained"}))
    finally:
        if env is not None:
            env.close()
    app.close()
    return 0
