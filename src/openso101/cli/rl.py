# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 rl ...` subcommands.

train --algo selects the RL algorithm:
- ppo: real PPO training via rsl_rl + BestCheckpointRunner.
- sac: deferred to sub-project E.
- ppo_lag / cpo / focops: NOT supported (safe-RL stays in safe_sim2real).
"""

from __future__ import annotations

import argparse


_SAFE_ALGOS = ("ppo_lag", "cpo", "focops")
_OPENSO101_ALGOS = ("ppo", "sac")
_ALL_ALGOS = _OPENSO101_ALGOS + _SAFE_ALGOS


def _cmd_train(args: argparse.Namespace) -> int:
    if args.algo in _SAFE_ALGOS:
        print(
            f"openso101 rl train: --algo {args.algo} is not available in "
            "OpenSO-101. Safe-RL stays in the legacy safe_sim2real repository."
        )
        return 2

    if args.algo == "sac":
        print(
            "openso101 rl train: --algo sac is part of sub-project E "
            "(off-policy RL) and not yet implemented in this refactor."
        )
        return 2

    if args.algo != "ppo":
        # Should be unreachable due to argparse choices.
        print(f"openso101 rl train: unknown algorithm {args.algo!r}")
        return 2

    # --- PPO training body (ported from
    # /data/safe_sim2real/src/safe_sim2real/scripts/rsl_rl/train.py) ---

    # AppLauncher must launch BEFORE any isaaclab / isaaclab_tasks / isaaclab_rl
    # / rsl_rl Isaac-Sim-bound imports.
    from isaaclab.app import AppLauncher

    # Vision tasks instantiate the SO-101 overhead and wrist cameras. State-only
    # tasks can run without the rendering pipeline.
    enable_cameras = bool(
        getattr(args, "video", False)
        or getattr(args, "with_cameras", False)
        or (args.task is not None and "-Vision" in args.task)
    )

    # The new CLI argparse only exposes `headless`; other legacy AppLauncher flags
    # (video recording via AppLauncher, distributed, multi-gpu, --device,
    # --enable_cameras) are intentionally dropped here and replaced with
    # sensible defaults. They can be re-added later if needed.
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=enable_cameras)
    simulation_app = app_launcher.app

    # --- minimum rsl_rl version check (mirrors legacy script) ---
    import importlib.metadata as metadata
    import platform

    from packaging import version

    RSL_RL_VERSION = "3.0.1"
    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
        if platform.system() == "Windows":
            cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        else:
            cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
        print(
            f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
            f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
            f"\n\n\t{' '.join(cmd)}\n"
        )
        return 1

    # --- Rest of imports (must follow AppLauncher) ---
    import gymnasium as gym
    import os
    from datetime import datetime

    import torch

    import omni
    from rsl_rl.runners import DistillationRunner

    from openso101.rl.runners.on_policy_runner import BestCheckpointRunner
    from openso101.rl import cli_args as _cli_args

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    import openso101.tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # Hydra needs to consume `agent` to pick the config entry point.
    agent_entry_point = getattr(args, "agent", "rsl_rl_cfg_entry_point")

    # The new CLI argparse for `train` doesn't expose the full rsl_rl arg group
    # (those flags live in openso101.rl.cli_args.add_rsl_rl_args for callers
    # who want them). Backfill the attributes that update_rsl_rl_cfg reads so
    # an `rl train` invocation works without --resume/--load_run/etc.
    for _attr, _default in (
        ("resume", False),
        ("load_run", None),
        ("checkpoint", None),
        ("run_name", None),
        ("logger", "wandb"),
        ("log_project_name", "openso101"),
    ):
        if not hasattr(args, _attr):
            setattr(args, _attr, _default)

    # Hydra reads sys.argv directly for override-style key=value flags
    # (e.g. agent.algorithm.gamma=0.95). The new CLI's argparse has already
    # consumed its own flags (--task/--algo/--headless/...) but they still
    # live in sys.argv, so Hydra would choke on them. Keep only Hydra-style
    # overrides for it to parse.
    import sys as _sys
    _sys.argv = [_sys.argv[0]] + [
        a for a in _sys.argv[1:] if "=" in a and not a.startswith("-")
    ]

    @hydra_task_config(args.task, agent_entry_point)
    def _run(env_cfg, agent_cfg):
        """Train with RSL-RL agent."""
        agent_cfg = _cli_args.update_rsl_rl_cfg(agent_cfg, args)

        # Apply variant hooks. The default is base RL (action_mode="rl",
        # cameras=False, play=False). --with-cameras adds wrist + overhead
        # cameras (needed for vision policies).
        if getattr(args, "with_cameras", False):
            env_cfg.configure_cameras(True)

        env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args.max_iterations if args.max_iterations is not None else agent_cfg.max_iterations
        )

        env_cfg.seed = agent_cfg.seed
        # The new CLI doesn't expose --device; leave env_cfg.sim.device at its default.

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"[INFO] Run timestamp: {run_stamp}")
        log_dir_name = run_stamp
        if agent_cfg.run_name:
            log_dir_name += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir_name)

        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            # The new CLI doesn't expose --export_io_descriptors; default to False.
            env_cfg.export_io_descriptors = False
            env_cfg.io_descriptors_output_dir = log_dir
        else:
            omni.log.warn(
                "IO descriptors are only supported for manager based RL environments. "
                "No IO descriptors will be exported."
            )

        env_cfg.log_dir = log_dir

        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        if args.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            # OpenSO-101 default: BestCheckpointRunner for model_best.pt tracking.
            runner = BestCheckpointRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.add_git_repo_to_log(__file__)
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        env.close()

    try:
        _run()
    finally:
        simulation_app.close()

    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    # AppLauncher must launch BEFORE any isaaclab / rsl_rl imports.
    from isaaclab.app import AppLauncher

    enable_cameras = bool(
        getattr(args, "with_cameras", False)
        or (args.task is not None and "-Vision" in args.task)
    )

    # Legacy play.py also accepted --video, --disable_fabric, --use_pretrained_checkpoint,
    # --real-time, --seed, --device. The new CLI doesn't expose those yet — intentionally
    # dropped here with sensible defaults. They can be added later if needed.
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=enable_cameras)
    simulation_app = app_launcher.app

    # --- Rest of imports (must follow AppLauncher) ---
    import gymnasium as gym
    import os
    import time

    import torch

    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from openso101.rl import cli_args as _cli_args

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

    import isaaclab_tasks  # noqa: F401
    import openso101.tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    agent_entry_point = getattr(args, "agent", "rsl_rl_cfg_entry_point")

    # The new CLI argparse for `play` doesn't ship the full rsl_rl arg group, so we
    # synthesize the namespace attributes that `update_rsl_rl_cfg` reads.
    for _attr, _default in (
        ("seed", None),
        ("resume", None),
        ("load_run", None),
        ("run_name", None),
        ("logger", None),
        ("log_project_name", None),
    ):
        if not hasattr(args, _attr):
            setattr(args, _attr, _default)

    # Strip CLI flags from sys.argv so Hydra only sees its own key=value overrides.
    import sys as _sys
    _sys.argv = [_sys.argv[0]] + [
        a for a in _sys.argv[1:] if "=" in a and not a.startswith("-")
    ]

    @hydra_task_config(args.task, agent_entry_point)
    def _run(env_cfg, agent_cfg):
        """Play with RSL-RL agent."""
        task_name = args.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = _cli_args.update_rsl_rl_cfg(agent_cfg, args)

        # Apply variant hooks: play mode (small env count, no DR), plus cameras
        # if the user asked. Hydra has already resolved overrides into env_cfg;
        # the variant hooks layer the play/cameras specialization on top.
        env_cfg.configure_play(True)
        if getattr(args, "with_cameras", False):
            env_cfg.configure_cameras(True)

        env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs

        env_cfg.seed = agent_cfg.seed

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args.checkpoint:
            resume_path = retrieve_file_path(args.checkpoint)
        else:
            try:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            except FileNotFoundError as err:
                print(f"[ERROR] No trained run found in: {log_root_path}")
                print("[ERROR] Train a policy first or provide --checkpoint.")
                print(f"[HINT] Example: openso101 rl train --task {train_task_name} --algo ppo --headless")
                raise SystemExit(1) from err

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        policy = runner.get_inference_policy(device=env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        obs = env.get_observations()
        while simulation_app.is_running():
            _ = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

        env.close()

    try:
        _run()
    finally:
        simulation_app.close()

    return 0


# --- Plot helpers (ported from
# /data/safe_sim2real/src/safe_sim2real/scripts/plot_training.py) ---


def _plot_smooth(y, window: int = 20):
    """Simple uniform moving-average smoother."""
    import numpy as np

    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def _plot_load_scalars(log_dir: str):
    """Return {tag: (steps, values)} for every scalar tag in the log dir."""
    import os

    from tensorboard.backend.event_processing import event_accumulator

    tf_path = next(
        (os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events")),
        None,
    )
    if tf_path is None:
        raise FileNotFoundError(f"No tfevents file found in {log_dir}")

    ea = event_accumulator.EventAccumulator(tf_path)
    ea.Reload()

    data: dict[str, tuple[list, list]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        data[tag] = (steps, vals)
    return data


def _plot_find_latest_log(task: str) -> str:
    import os

    root = os.path.join("logs", "rsl_rl", task)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No logs found at {root}")
    runs = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No runs found in {root}")
    return os.path.join(root, runs[0])


_STAGE_COLORS = {
    # atomic/lift (Isaac Lab official lift design)
    "reaching_object": "#4e9af1",
    "lifting_object": "#f1c14e",
    "object_goal_tracking": "#a84ef1",
    "object_goal_tracking_fine_grained": "#4ef1a8",
    "stage_0_lift_goal": "#5fae5f",
    "stage_0_complete_bonus": "#347d39",
    "stage_1_carry_goal": "#4e9af1",
    "stage_1_complete_bonus": "#225ea8",
    "stage_2_place_goal": "#a84ef1",
    # smoothness (shared)
    "action_rate": "#aaaaaa",
    "joint_vel": "#cccccc",
}

_STAGE_LABELS = {
    "reaching_object": "Reaching object",
    "lifting_object": "Lifting (z > 0.04)",
    "object_goal_tracking": "Goal tracking (std=0.3)",
    "object_goal_tracking_fine_grained": "Goal tracking (std=0.05)",
    "stage_0_lift_goal": "Stage 0: lifted cube -> lift ball",
    "stage_0_complete_bonus": "Stage 0 touched",
    "stage_1_carry_goal": "Stage 1: carried cube -> carry ball",
    "stage_1_complete_bonus": "Stage 1 touched",
    "stage_2_place_goal": "Stage 2: cube -> place ball",
    "action_rate": "Action rate penalty",
    "joint_vel": "Joint vel penalty",
}


def _plot_run(log_dir: str, save: bool = False, smooth_window: int = 30) -> None:
    import os

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    data = _plot_load_scalars(log_dir)
    run_name = os.path.basename(log_dir)
    task_name = os.path.basename(os.path.dirname(log_dir))

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Training curves - {task_name}  [{run_name}]", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, :2])
    if "Train/mean_reward" in data:
        steps, vals = data["Train/mean_reward"]
        ax.plot(steps, vals, alpha=0.25, color="#555", linewidth=0.8)
        ax.plot(steps, _plot_smooth(np.array(vals), smooth_window), color="#222", linewidth=2,
                label="mean reward (smoothed)")
    ax.set_title("Total episode reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    if "Train/mean_episode_length" in data:
        steps, vals = data["Train/mean_episode_length"]
        ax2.plot(steps, _plot_smooth(np.array(vals), smooth_window), color="#2266cc", linewidth=2)
    ax2.set_title("Mean episode length")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :2])
    stage_keys = [
        "reaching_object",
        "lifting_object",
        "object_goal_tracking",
        "object_goal_tracking_fine_grained",
        "stage_0_lift_goal",
        "stage_0_complete_bonus",
        "stage_1_carry_goal",
        "stage_1_complete_bonus",
        "stage_2_place_goal",
    ]
    all_steps = None
    stage_arrays = {}
    for key in stage_keys:
        tag = f"Episode_Reward/{key}"
        if tag in data:
            steps, vals = data[tag]
            arr = _plot_smooth(np.array(vals), smooth_window)
            stage_arrays[key] = (steps, np.clip(arr, 0, None))
            if all_steps is None:
                all_steps = steps

    if all_steps is not None:
        n = len(all_steps)
        baseline = np.zeros(n)
        for key in stage_keys:
            if key in stage_arrays:
                steps_k, arr_k = stage_arrays[key]
                ax3.fill_between(steps_k, baseline, baseline + arr_k,
                                 alpha=0.7, color=_STAGE_COLORS[key], label=_STAGE_LABELS[key])
                baseline = baseline + arr_k

    ax3.set_title("Reward breakdown by stage (stacked, smoothed)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Reward contribution")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    for key in stage_keys:
        tag = f"Episode_Reward/{key}"
        if tag in data:
            steps, vals = data[tag]
            ax4.plot(steps, _plot_smooth(np.array(vals), smooth_window),
                     color=_STAGE_COLORS[key], label=_STAGE_LABELS[key], linewidth=1.5)
    ax4.set_title("Per-stage reward (individual)")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Reward")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    if "Policy/mean_noise_std" in data:
        steps, vals = data["Policy/mean_noise_std"]
        ax5.plot(steps, vals, color="#aa4400", linewidth=1.5)
        ax5.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, label="std=0.1 (collapsed)")
    ax5.set_title("Policy noise std\n(should stay > 0.1)")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("std")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    for tag, label, color in [
        ("Loss/surrogate", "Surrogate (policy)", "#4477aa"),
        ("Loss/value_function", "Value function", "#aa4477"),
        ("Loss/entropy", "Entropy", "#44aa77"),
    ]:
        if tag in data:
            steps, vals = data[tag]
            ax6.plot(steps, _plot_smooth(np.array(vals), smooth_window),
                     label=label, color=color, linewidth=1.5)
    ax6.set_title("PPO losses")
    ax6.set_xlabel("Iteration")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 2])
    term_tags = {
        "Episode_Termination/time_out": ("Timeout", "#aaaaaa"),
        "Episode_Termination/object_dropping": ("Dropped", "#bb4444"),
    }
    for tag, (label, color) in term_tags.items():
        if tag in data:
            steps, vals = data[tag]
            ax7.plot(steps, _plot_smooth(np.array(vals), smooth_window),
                     label=label, color=color, linewidth=1.5)
    ax7.set_title("Termination rates")
    ax7.set_xlabel("Iteration")
    ax7.set_ylabel("Fraction of episodes")
    ax7.set_ylim(0, 1.05)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out_path = os.path.join(log_dir, "training_curves.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved -> {out_path}")
    else:
        plt.show()


def _cmd_plot(args: argparse.Namespace) -> int:
    # `--run` is an alias for `--log_dir`.
    log_dir = getattr(args, "log_dir", None) or getattr(args, "run", None)
    if log_dir is None:
        log_dir = _plot_find_latest_log(args.task)
    print(f"[INFO] Reading from: {log_dir}")
    _plot_run(log_dir, save=args.save, smooth_window=args.smooth)
    return 0


def add_subparsers(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="rl_cmd", required=True)

    p_train = sub.add_parser("train", help="Train an RL policy")
    p_train.add_argument("--task", required=True, help="Gym ID")
    p_train.add_argument(
        "--algo",
        required=True,
        choices=_ALL_ALGOS,
        help="Algorithm (ppo|sac|ppo_lag|cpo|focops)",
    )
    p_train.add_argument("--num_envs", type=int, default=None)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--max_iterations", type=int, default=None)
    p_train.add_argument("--video", action="store_true")
    p_train.add_argument("--video_length", type=int, default=200)
    p_train.add_argument("--video_interval", type=int, default=2400)
    p_train.add_argument("--with-cameras", action="store_true")
    p_train.add_argument("--headless", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_play = sub.add_parser("play", help="Replay a trained checkpoint")
    p_play.add_argument("--task", required=True)
    p_play.add_argument("--checkpoint", required=True)
    p_play.add_argument("--num_envs", type=int, default=None)
    p_play.add_argument("--with-cameras", action="store_true")
    p_play.add_argument("--headless", action="store_true")
    p_play.set_defaults(func=_cmd_play)

    p_plot = sub.add_parser("plot", help="Plot training curves from a run dir")
    group = p_plot.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Exact run directory path (alias for --log_dir)")
    group.add_argument("--log_dir", help="Exact run directory path")
    group.add_argument("--task", help="Task name (auto-picks latest run)")
    p_plot.add_argument("--smooth", type=int, default=30)
    p_plot.add_argument("--save", action="store_true")
    p_plot.set_defaults(func=_cmd_plot)
