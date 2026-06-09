# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""`openso101 rl ...` subcommands.

train --algo selects the RL algorithm:
- ppo:          on-policy PPO via rsl_rl + BestCheckpointRunner.
- distillation: teacher → student knowledge transfer via rsl_rl
                DistillationRunner; requires --teacher-checkpoint.
"""

from __future__ import annotations

import argparse


_OPENSO101_ALGOS = ("ppo", "distillation")
_ALL_ALGOS = _OPENSO101_ALGOS

# Per-algo `gym.make` agent-entry-point key. Tasks register a config
# class under each key they support; the CLI looks up the right one based
# on `--algo`.
_ALGO_TO_ENTRY_POINT = {
    "ppo": "rsl_rl_cfg_entry_point",
    "distillation": "rsl_rl_distillation_cfg_entry_point",
}


def _cmd_train(args: argparse.Namespace) -> int:
    if args.algo not in _OPENSO101_ALGOS:
        # Should be unreachable due to argparse choices.
        print(f"openso101 rl train: unknown algorithm {args.algo!r}")
        return 2

    if args.algo == "distillation":
        teacher = getattr(args, "teacher_checkpoint", None)
        if not teacher:
            print(
                "openso101 rl train --algo distillation: --teacher-checkpoint "
                "is required. Point it at a PPO run directory (e.g. "
                "`logs/rsl_rl/pick_place/2026-05-14_12-30-00`) or a direct "
                ".pt file. The student is trained to mimic the teacher's "
                "action distribution."
            )
            return 2

    # --- PPO training body (ported from the predecessor safe_sim2real
    # project's rsl_rl training script) ---

    # AppLauncher must launch BEFORE any isaaclab / isaaclab_tasks / isaaclab_rl
    # / rsl_rl Isaac-Sim-bound imports.
    from isaaclab.app import AppLauncher

    # Vision tasks instantiate the SO-101 overhead and wrist cameras. State-only
    # tasks can run without the rendering pipeline.
    #
    # `--visual-dr` also needs the render pipeline even without cameras:
    # visual DR writes USD attributes (light intensity / color, object
    # material) on every reset, and those writes need Fabric to propagate.
    # Without an initialized RTX pipeline the writes queue up indefinitely
    # and the very first `env.reset()` hangs silently in the runner. Cost
    # of the pipeline init is ~0.5-1 GB VRAM, NOT multiplicative with
    # num_envs (unlike actual camera observation rendering), so this is
    # safe on small GPUs as long as the user doesn't also pass
    # `--with-cameras`.
    enable_cameras = bool(
        getattr(args, "video", False)
        or getattr(args, "with_cameras", False)
        or getattr(args, "visual_dr", False)
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
    # `--algo distillation` flips the lookup to `rsl_rl_distillation_cfg_entry_point`
    # so we pull the StudentTeacher cfg instead of the PPO actor-critic cfg.
    agent_entry_point = getattr(args, "agent", None) or _ALGO_TO_ENTRY_POINT[args.algo]

    # The new CLI argparse for `train` doesn't expose the full rsl_rl arg group
    # (those flags live in openso101.rl.cli_args.add_rsl_rl_args for callers
    # who want them). Backfill the attributes that update_rsl_rl_cfg reads so
    # an `rl train` invocation works without --resume/--load_run/etc.
    #
    # IMPORTANT: only backfill the default when the attribute is ABSENT. The
    # train subparser now exposes --resume / --load_run / --checkpoint directly,
    # so when the user passes them we must honor their values (previously this
    # block hardcoded resume=False/load_run=None/checkpoint=None even though the
    # _run closure fully supports resume). `run_name` / `logger` /
    # `log_project_name` are likewise only filled when the parser didn't.
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

    # Treat an explicit --load_run / --checkpoint as a resume request even if
    # the user forgot --resume; otherwise update_rsl_rl_cfg would stage the
    # checkpoint but the _run closure's `if agent_cfg.resume` guard would skip
    # loading it. (Distillation reuses load_run/checkpoint without resume, so
    # only flip this on for the on-policy path.)
    if args.algo == "ppo" and (
        getattr(args, "load_run", None) or getattr(args, "checkpoint", None)
    ):
        args.resume = True

    # Auto-fallback when the user picked wandb but it isn't importable in
    # this env (wandb lives in [project.optional-dependencies] and a fresh
    # scripts/install.sh does not pull it). Falling through to wandb here
    # would crash rsl_rl deep in the runner on `import wandb`.
    if args.logger == "wandb":
        try:
            import wandb  # noqa: F401
        except ImportError:
            print(
                "[WARN] --logger wandb requested but wandb is not installed. "
                "Falling back to tensorboard. Install with `pip install wandb` "
                "and re-run for browser-based monitoring."
            )
            args.logger = "tensorboard"
        else:
            print(
                f"[INFO] Logging to W&B project '{args.log_project_name}'. "
                "If WANDB_API_KEY is unset, wandb will prompt or fall back to "
                "anonymous mode."
            )

    # Distillation: the teacher checkpoint flag is the only user-facing
    # way to point at a teacher. Internally rsl_rl reads it from
    # `agent_cfg.load_run` + `agent_cfg.load_checkpoint`, which our existing
    # `update_rsl_rl_cfg` pulls from `args.load_run` / `args.checkpoint`.
    # Translate `--teacher-checkpoint` into both: if the user gave a directory,
    # use it as `load_run`; if they gave a file path, split into parent + name.
    if args.algo == "distillation":
        from pathlib import Path as _Path
        teacher = _Path(args.teacher_checkpoint).expanduser().resolve()
        if not teacher.exists():
            raise FileNotFoundError(
                f"--teacher-checkpoint not found: {teacher}"
            )
        if teacher.is_dir():
            args.load_run = str(teacher)
            args.checkpoint = None
        else:
            args.load_run = str(teacher.parent)
            args.checkpoint = teacher.name

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
        # cameras (needed for vision policies). --visual-dr randomizes
        # light + object color at each reset for sim2real perception
        # transfer.
        if getattr(args, "with_cameras", False):
            env_cfg.configure_cameras(True)
        if getattr(args, "visual_dr", False):
            if hasattr(env_cfg, "configure_visual_dr"):
                env_cfg.configure_visual_dr(True)
            else:
                print(
                    "[WARN]: --visual-dr requested but this task lacks a "
                    "configure_visual_dr() hook; skipping."
                )

        env_cfg.scene.num_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args.max_iterations if args.max_iterations is not None else agent_cfg.max_iterations
        )

        # Seed threading: update_rsl_rl_cfg() above already copied args.seed
        # onto agent_cfg.seed (resolving --seed -1 to a fresh random seed and
        # leaving agent_cfg.seed at its config default when --seed was omitted).
        # Propagate that single source of truth onto the env so the sim RNG and
        # the policy RNG agree for reproducible runs.
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
    except BaseException as exc:
        # Print the traceback to fd 1 BEFORE the finally clause runs.
        # `simulation_app.close()` can take 30-60s on shutdown and Python
        # buffers stderr aggressively in that window, so without this the
        # actual exception message gets eaten if shutdown stalls.
        import os as _os
        import traceback as _tb
        _os.write(1, f"\n[rl train] crashed: {exc!r}\n".encode())
        _os.write(1, _tb.format_exc().encode())
        raise
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
    import sys
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
        # OpenSO-101 uses one gym ID per task (no '-Play' suffix variant —
        # play mode is selected via configure_play below), so the task
        # name is also the training task name.
        task_name = args.task.split(":")[-1]
        train_task_name = task_name

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
            # Task-identity check: refuse to load a checkpoint from a
            # different task's experiment dir even when shapes happen to
            # match. The runner cfg's experiment_name is task-specific
            # (pick_place / lift / stack), so a checkpoint whose ancestor
            # path doesn't contain `/rsl_rl/<expected>/` is cross-task
            # and will silently misbehave.
            expected = f"/rsl_rl/{agent_cfg.experiment_name}/"
            if expected not in os.path.abspath(resume_path):
                raise SystemExit(
                    f"\n[ERROR]: Checkpoint is from a different task.\n"
                    f"  --task             : {args.task}  (experiment {agent_cfg.experiment_name!r})\n"
                    f"  --checkpoint       : {resume_path}\n"
                    f"OpenSO-101 task policies are NOT cross-task transferable; "
                    f"even when observation dims happen to match, the semantics "
                    f"of each observation term (curriculum stage, goal pose, ...) "
                    f"differ between tasks. Re-run with a checkpoint from "
                    f"logs/rsl_rl/{agent_cfg.experiment_name}/...\n"
                )
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
        try:
            runner.load(resume_path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise SystemExit(
                    f"\n[ERROR]: Checkpoint shape mismatch — the checkpoint at\n"
                    f"  {resume_path}\n"
                    f"was trained for a different task than {task_name!r}.\n"
                    f"Checkpoints are NOT transferable across OpenSO-101 tasks: each "
                    f"task has its own observation and action space (e.g. PickPlace "
                    f"includes a 3-stage curriculum goal that Lift/Stack don't). Use "
                    f"a checkpoint produced by 'openso101 rl train --task {task_name}'.\n"
                ) from e
            raise

        # get_inference_policy() returns policy.act_inference, which evaluates
        # the actor MLP and returns the action MEAN (no Normal sampling). So
        # play is deterministic by construction; --deterministic is accepted as
        # an explicit no-op for parity with `rl eval` and does not change this
        # code path (there is no fresh sample to suppress).
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

        # Heartbeat print so users know the play loop is stepping (Isaac
        # Sim's stdio hijack tends to swallow the otherwise-silent inner
        # loop and makes the process look hung).
        print(
            f"[INFO]: Play loop running. Stepping policy at dt={dt:.4f}s. "
            f"Close the Isaac Sim window or Ctrl+C to stop.",
            flush=True,
        )

        obs = env.get_observations()
        step_count = 0
        last_heartbeat = time.time()
        while simulation_app.is_running():
            _ = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            step_count += 1
            # Heartbeat every ~5 s of wall time.
            now = time.time()
            if now - last_heartbeat > 5.0:
                sys.__stdout__.write(f"[INFO]: play step {step_count}\n")
                sys.__stdout__.flush()
                last_heartbeat = now

        env.close()

    try:
        _run()
    finally:
        simulation_app.close()

    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Quantitative checkpoint eval: success rate + a reach/grasp/lift funnel.

    Mirrors _cmd_play's env-construction and checkpoint-loading path (same
    configure_play, same task-identity guard, same runner.load), then steps the
    DETERMINISTIC inference policy (runner.get_inference_policy() returns the
    actor's act_inference, i.e. the action mean — no fresh Normal sample) under
    ManagerBasedRLEnv auto-reset until every env has finished its quota of
    episodes. Per env we latch booleans across each episode by reading the
    'success' termination term, the contact-confirmed grasp, and the object
    lift height, then aggregate the completed episodes into a JSON report.
    """
    from isaaclab.app import AppLauncher

    enable_cameras = bool(
        getattr(args, "with_cameras", False)
        or (args.task is not None and "-Vision" in args.task)
    )
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=enable_cameras)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import json
    import math
    import os

    import torch

    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from openso101.rl import cli_args as _cli_args
    from openso101.tasks.shared.grasp import object_grasped_by_jaws

    from isaaclab.envs import (
        DirectMARLEnv,
        ManagerBasedRLEnvCfg,  # noqa: F401
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    import openso101.tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    agent_entry_point = getattr(args, "agent", "rsl_rl_cfg_entry_point")

    # Fixed seed for determinism (sim RNG is seeded via env_cfg.seed below; this
    # covers any torch-side RNG the policy/normalizer might touch).
    torch.manual_seed(args.seed)

    # update_rsl_rl_cfg reads these; synthesize the ones the eval parser omits.
    for _attr, _default in (
        ("resume", None),
        ("load_run", None),
        ("run_name", None),
        ("logger", None),
        ("log_project_name", None),
    ):
        if not hasattr(args, _attr):
            setattr(args, _attr, _default)

    import sys as _sys
    _sys.argv = [_sys.argv[0]] + [
        a for a in _sys.argv[1:] if "=" in a and not a.startswith("-")
    ]

    @hydra_task_config(args.task, agent_entry_point)
    def _run(env_cfg, agent_cfg):
        task_name = args.task.split(":")[-1]

        agent_cfg = _cli_args.update_rsl_rl_cfg(agent_cfg, args)

        # Same play specialization (small-but-not-tiny env count is fine; we
        # override num_envs explicitly below) and no domain randomization.
        env_cfg.configure_play(True)
        if getattr(args, "with_cameras", False):
            env_cfg.configure_cameras(True)

        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed

        # Same task-identity guard play uses: refuse a checkpoint from another
        # task's experiment dir even if observation/action shapes happen to
        # match (per-term observation semantics differ across tasks).
        resume_path = retrieve_file_path(args.checkpoint)
        expected = f"/rsl_rl/{agent_cfg.experiment_name}/"
        if expected not in os.path.abspath(resume_path):
            raise SystemExit(
                f"\n[ERROR]: Checkpoint is from a different task.\n"
                f"  --task       : {args.task}  (experiment {agent_cfg.experiment_name!r})\n"
                f"  --checkpoint : {resume_path}\n"
                f"OpenSO-101 task policies are NOT cross-task transferable. "
                f"Use a checkpoint from logs/rsl_rl/{agent_cfg.experiment_name}/...\n"
            )

        env_cfg.log_dir = os.path.dirname(resume_path)
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        try:
            runner.load(resume_path)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise SystemExit(
                    f"\n[ERROR]: Checkpoint shape mismatch — {resume_path} was "
                    f"trained for a different task than {task_name!r}. "
                    f"Checkpoints are NOT transferable across OpenSO-101 tasks.\n"
                ) from e
            raise

        # Deterministic policy: act_inference == action mean (no sampling).
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        n_envs = env.num_envs
        device = env.unwrapped.device
        unwrapped = env.unwrapped
        term_mgr = unwrapped.termination_manager
        has_success_term = "success" in term_mgr.active_terms

        episodes_per_env = math.ceil(args.n_episodes / n_envs)
        target_total = episodes_per_env * n_envs

        # Per-env latches (reset on each episode boundary).
        ever_succeeded = torch.zeros(n_envs, dtype=torch.bool, device=device)
        ever_grasped = torch.zeros(n_envs, dtype=torch.bool, device=device)
        ever_lifted = torch.zeros(n_envs, dtype=torch.bool, device=device)
        ever_reached = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_success_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        steps_in_episode = torch.zeros(n_envs, dtype=torch.long, device=device)

        # Per-env completed-episode counters and aggregate accumulators.
        completed = torch.zeros(n_envs, dtype=torch.long, device=device)
        agg = {
            "success": 0,
            "grasped": 0,
            "lifted": 0,
            "reached": 0,
            "dropped": 0,
            "timeout": 0,
            "steps_to_success_sum": 0,
            "steps_to_success_n": 0,
            "total": 0,
        }

        has_drop_term = "object_dropping" in term_mgr.active_terms
        reach_threshold = 0.05  # m, ee->object; optional funnel stage
        lift_threshold = 0.04   # m above the env origin, "off the table"

        obj = unwrapped.scene["object"]
        env_origin_z = unwrapped.scene.env_origins[:, 2]

        print(
            f"[INFO]: Evaluating {agent_cfg.experiment_name} over "
            f"{target_total} episodes ({n_envs} envs x {episodes_per_env}).",
            flush=True,
        )

        obs = env.get_observations()
        while int(completed.min().item()) < episodes_per_env:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)

            steps_in_episode += 1

            # --- Latch funnel signals from THIS step (pre-reset state). ---
            grasped_now = object_grasped_by_jaws(unwrapped)
            ever_grasped |= grasped_now

            lifted_now = (obj.data.root_pos_w[:, 2] - env_origin_z) > lift_threshold
            ever_lifted |= lifted_now

            # Optional reach stage: ee->object distance. Best-effort — if the
            # observation doesn't expose it cheaply, fall back to the grasp
            # sensors' parent bodies via the cube being close to a jaw is
            # already captured by grasp; here we approximate "reached" as
            # "grasped or lifted or very close" using cube-to-origin proxy is
            # unreliable, so we tie reach to a small ee->obj distance when the
            # task exposes an 'ee_frame' transform; otherwise reach == grasped.
            try:
                ee_frame = unwrapped.scene["ee_frame"]
                ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
                ee_obj_dist = torch.linalg.vector_norm(
                    ee_pos_w - obj.data.root_pos_w, dim=-1
                )
                ever_reached |= ee_obj_dist < reach_threshold
            except (KeyError, AttributeError, IndexError):
                ever_reached |= grasped_now

            if has_success_term:
                succ_now = term_mgr.get_term("success")
                newly = succ_now & ~ever_succeeded
                first_success_step[newly] = steps_in_episode[newly]
                ever_succeeded |= succ_now

            # --- Record outcomes for envs that ended this step. ---
            done_mask = dones.to(torch.bool)
            if bool(done_mask.any()):
                idx = torch.nonzero(done_mask, as_tuple=False).flatten()
                # Only count episodes up to the quota per env.
                for i in idx.tolist():
                    if completed[i] >= episodes_per_env:
                        continue
                    completed[i] += 1
                    agg["total"] += 1
                    if bool(ever_succeeded[i]):
                        agg["success"] += 1
                        if first_success_step[i] >= 0:
                            agg["steps_to_success_sum"] += int(first_success_step[i].item())
                            agg["steps_to_success_n"] += 1
                    if bool(ever_grasped[i]):
                        agg["grasped"] += 1
                    if bool(ever_lifted[i]):
                        agg["lifted"] += 1
                    if bool(ever_reached[i]):
                        agg["reached"] += 1
                    # Timeout vs drop attribution from termination terms /
                    # time_outs. A success that coincides with timeout still
                    # counts as success in the success column above.
                    is_timeout = bool(extras.get("time_outs", torch.zeros(n_envs, dtype=torch.bool, device=device))[i])
                    if has_drop_term and bool(term_mgr.get_term("object_dropping")[i]):
                        agg["dropped"] += 1
                    elif is_timeout and not bool(ever_succeeded[i]):
                        agg["timeout"] += 1

                # Clear latches for the envs that just auto-reset.
                ever_succeeded[done_mask] = False
                ever_grasped[done_mask] = False
                ever_lifted[done_mask] = False
                ever_reached[done_mask] = False
                first_success_step[done_mask] = -1
                steps_in_episode[done_mask] = 0

        # --- Aggregate ---
        total = max(agg["total"], 1)
        success_rate = agg["success"] / total
        # Wald 95% CI on the success-rate proportion.
        ci95 = 1.96 * math.sqrt(max(success_rate * (1 - success_rate), 0.0) / total)
        mean_steps_to_success = (
            agg["steps_to_success_sum"] / agg["steps_to_success_n"]
            if agg["steps_to_success_n"] > 0
            else None
        )
        report = {
            "task": task_name,
            "checkpoint": os.path.abspath(resume_path),
            "n_episodes": agg["total"],
            "success_rate": success_rate,
            "success_rate_ci95": ci95,
            "mean_steps_to_success": mean_steps_to_success,
            "drop_rate": agg["dropped"] / total,
            "timeout_rate": agg["timeout"] / total,
            "frac_reached": agg["reached"] / total,
            "frac_grasped": agg["grasped"] / total,
            "frac_lifted": agg["lifted"] / total,
            "seed": args.seed,
            "num_envs": n_envs,
        }
        if not has_success_term:
            report["warning"] = (
                "no 'success' termination term active on this task; "
                "success_rate is 0 and not meaningful."
            )

        print("\n===== eval report =====", flush=True)
        for k, v in report.items():
            print(f"  {k}: {v}", flush=True)

        out_path = os.path.join(os.path.dirname(resume_path), "eval_report.json")
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[INFO] Saved -> {out_path}", flush=True)

        env.close()

    try:
        _run()
    finally:
        simulation_app.close()

    return 0


# --- Plot helpers (ported from the predecessor safe_sim2real project's
# training-plot script) ---


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
    # pick-and-lift (sentinel delta shaping)
    "pregrasp_approach": "#5fae5f",
    "grasp_hold": "#f1c14e",
    "carry_to_goal": "#4e9af1",
    "success_bonus": "#a84ef1",
    # smoothness (shared)
    "action_rate": "#aaaaaa",
    "joint_vel": "#cccccc",
}

_STAGE_LABELS = {
    "reaching_object": "Reaching object",
    "lifting_object": "Lifting (z > 0.04)",
    "object_goal_tracking": "Goal tracking (std=0.3)",
    "object_goal_tracking_fine_grained": "Goal tracking (std=0.05)",
    "pregrasp_approach": "Pregrasp approach (delta eef->obj)",
    "grasp_hold": "Grasp hold (contact-confirmed)",
    "carry_to_goal": "Carry to goal (delta obj->goal)",
    "success_bonus": "Success (in goal AND grasped)",
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

    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(f"Training curves - {task_name}  [{run_name}]", fontsize=13, fontweight="bold")

    # 4 rows now: rows 0-2 keep the original panels, row 3 adds the
    # success-rate-over-training curve and the goal-distance metric panel.
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

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
        "pregrasp_approach",
        "grasp_hold",
        "carry_to_goal",
        "success_bonus",
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

    # --- Success rate over training (row 3, spans 2 cols) ---
    # Auto-discover every Episode_Termination/* tag whose name mentions
    # 'success'. The shared RL contract renames each task's success
    # termination attr to 'success', so the tag is Episode_Termination/success;
    # discovering by substring keeps this robust to any legacy aliases
    # (lift_success / stacked_success) still present in older event files.
    ax8 = fig.add_subplot(gs[3, :2])
    success_tags = sorted(
        tag
        for tag in data
        if tag.startswith("Episode_Termination/") and "success" in tag.lower()
    )
    _success_palette = ["#2ca25f", "#1b7837", "#66c2a4", "#006d2c"]
    for i, tag in enumerate(success_tags):
        steps, vals = data[tag]
        label = tag.split("/", 1)[1]
        ax8.plot(
            steps,
            _plot_smooth(np.array(vals), smooth_window),
            color=_success_palette[i % len(_success_palette)],
            linewidth=2,
            label=label,
        )
    if not success_tags:
        ax8.text(
            0.5, 0.5, "no Episode_Termination/*success* scalar logged yet",
            ha="center", va="center", transform=ax8.transAxes, fontsize=9, color="#999",
        )
    ax8.set_title("Success rate over training\n(Episode_Termination/success, fraction of episodes)")
    ax8.set_xlabel("Iteration")
    ax8.set_ylabel("Success fraction")
    ax8.set_ylim(0, 1.05)
    if success_tags:
        ax8.legend(fontsize=8, loc="upper left")
    ax8.grid(True, alpha=0.3)

    # --- Distance-to-goal metric (row 3, last col) ---
    ax9 = fig.add_subplot(gs[3, 2])
    dist_tags = [
        ("Metrics/object_pose/distance_to_goal", "center distance", "#cc5500"),
        ("Metrics/object_pose/surface_distance_to_goal", "surface distance", "#0077aa"),
    ]
    plotted_dist = False
    for tag, label, color in dist_tags:
        if tag in data:
            steps, vals = data[tag]
            ax9.plot(steps, _plot_smooth(np.array(vals), smooth_window),
                     label=label, color=color, linewidth=1.5)
            plotted_dist = True
    if not plotted_dist:
        ax9.text(
            0.5, 0.5, "no Metrics/object_pose/distance_to_goal scalar",
            ha="center", va="center", transform=ax9.transAxes, fontsize=8, color="#999",
        )
    ax9.set_title("Object -> goal distance\n(lower is better)")
    ax9.set_xlabel("Iteration")
    ax9.set_ylabel("Distance (m)")
    if plotted_dist:
        ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

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
        help="Algorithm (ppo|distillation)",
    )
    p_train.add_argument(
        "--teacher-checkpoint",
        default=None,
        help=(
            "Teacher policy checkpoint for --algo distillation. Accepts "
            "either a directory (uses its `model_best.pt` / latest) or a "
            "direct .pt path. Required when --algo=distillation, ignored "
            "otherwise."
        ),
    )
    p_train.add_argument("--num_envs", type=int, default=None)
    p_train.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed for both the sim and the policy. Pass -1 for a fresh "
            "random seed each run; omit to use the agent config's default."
        ),
    )
    p_train.add_argument("--max_iterations", type=int, default=None)
    # Resume / fine-tune from an existing run. The _run closure already honors
    # agent_cfg.resume (it loads the checkpoint before runner.learn); these
    # flags just stop the namespace backfill from hardcoding resume off.
    p_train.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from a checkpoint (uses --load_run / --checkpoint).",
    )
    p_train.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run folder (name or path) to resume from. Implies --resume semantics.",
    )
    p_train.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file name within the resumed run (e.g. model_best.pt).",
    )
    # Video defaults to ON so the user can monitor headless training without
    # opening the simulator. Pass --no-video to disable. Recording a 200-step
    # clip every 2400 steps adds ~1 GB VRAM (RTX pipeline init) but no
    # significant CPU/GPU cost during gaps.
    p_train.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p_train.add_argument("--video_length", type=int, default=200)
    p_train.add_argument("--video_interval", type=int, default=2400)
    # Default to wandb so live metrics are visible in the browser without
    # tunneling tensorboard. Falls back to tensorboard if wandb isn't
    # installed (it's in [project.optional-dependencies] — `pip install
    # openso101[wandb]` or `pip install wandb`).
    p_train.add_argument(
        "--logger",
        choices=("wandb", "tensorboard", "neptune"),
        default="wandb",
        help="Where to log training metrics. Default 'wandb'.",
    )
    p_train.add_argument(
        "--log_project_name",
        default="openso101",
        help="W&B/Neptune project name. Ignored for tensorboard.",
    )
    p_train.add_argument("--with-cameras", action="store_true")
    p_train.add_argument(
        "--visual-dr",
        action="store_true",
        help=(
            "Randomize dome-light intensity + color and object color at each "
            "episode reset. Cheap sim2real perception transfer. Requires the "
            "task to expose a configure_visual_dr() hook."
        ),
    )
    p_train.add_argument("--headless", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_play = sub.add_parser("play", help="Replay a trained checkpoint")
    p_play.add_argument("--task", required=True)
    p_play.add_argument("--checkpoint", required=True)
    p_play.add_argument("--num_envs", type=int, default=None)
    p_play.add_argument("--with-cameras", action="store_true")
    p_play.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Step the deterministic inference policy (action mean) instead of "
            "sampling. This is already the default behaviour — runner."
            "get_inference_policy() returns act_inference, the action mean — so "
            "the flag is provided for explicitness/parity with eval."
        ),
    )
    p_play.add_argument("--headless", action="store_true")
    p_play.set_defaults(func=_cmd_play)

    p_eval = sub.add_parser(
        "eval",
        help="Quantitatively evaluate a checkpoint (success rate + funnel)",
    )
    p_eval.add_argument("--task", required=True, help="Gym ID")
    p_eval.add_argument("--checkpoint", required=True, help="Checkpoint .pt path")
    p_eval.add_argument("--num-envs", dest="num_envs", type=int, default=64)
    p_eval.add_argument("--n-episodes", dest="n_episodes", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--headless", action="store_true")
    p_eval.set_defaults(func=_cmd_eval)

    p_plot = sub.add_parser("plot", help="Plot training curves from a run dir")
    group = p_plot.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Exact run directory path (alias for --log_dir)")
    group.add_argument("--log_dir", help="Exact run directory path")
    group.add_argument("--task", help="Task name (auto-picks latest run)")
    p_plot.add_argument("--smooth", type=int, default=30)
    p_plot.add_argument("--save", action="store_true")
    p_plot.set_defaults(func=_cmd_plot)
