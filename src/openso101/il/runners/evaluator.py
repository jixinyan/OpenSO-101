import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

from openso101.il.runtime import _launch_isaac_app, resolve_policy_path, validate_positive_count
from openso101.scenes.models import file_digest
from openso101.il.policies.simulation import load_simulation_settings, apply_simulation_settings


def configure_evaluation(cfg, *, environments: int, seconds: float, scene=None, settings=None):
    from isaaclab.envs.mdp import time_out
    from isaaclab.managers import TerminationTermCfg
    from openso101.teleop.success import task_success_vector

    cfg.configure_action_mode("teleop")
    cfg.configure_cameras(True)
    if scene is not None:
        cfg.configure_scene(scene)
    apply_simulation_settings(cfg, settings)
    cfg.scene.num_envs = environments
    cfg.episode_length_s = seconds
    if hasattr(cfg.scene, "ee_frame") and cfg.scene.ee_frame is not None:
        cfg.scene.ee_frame.debug_vis = False
    success = TerminationTermCfg(func=task_success_vector)
    if getattr(cfg, "scene_spec", None) is not None:
        from openso101.scenes.isaaclab.runtime import task_failure

        failure = TerminationTermCfg(func=task_failure)
    else:
        from isaaclab.envs.mdp import root_height_below_minimum
        from isaaclab.managers import SceneEntityCfg

        objects = ("cube_top", "cube_bottom") if hasattr(cfg.scene, "cube_top") else ("object",)
        failure = {name: TerminationTermCfg(func=root_height_below_minimum,
                    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg(name)}) for name in objects}
        if hasattr(cfg.commands, "object_pose") and hasattr(cfg.commands.object_pose, "lock_stage"):
            from openso101.tasks.pick_place.mdp.terminations import StablePlacementSuccess

            cfg.current_step_placement_success = True
            success = TerminationTermCfg(func=StablePlacementSuccess, params={"settle_seconds": 0.5})
    failures = failure if isinstance(failure, dict) else {"failure": failure}
    cfg.terminations = {"time_out": TerminationTermCfg(func=time_out, time_out=True), **failures, "success": success}
    cfg.recorders = evaluation_recorder()


def evaluation_recorder():
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils import configclass

    class EvaluationRecorder(RecorderTerm):
        def record_post_step(self):
            env = self._env
            env._il_eval_transition = {
                "success": (env.termination_manager.get_term("success") & env.reset_buf).detach().clone(),
                "terminated": env.reset_terminated.detach().clone(),
                "truncated": env.reset_time_outs.detach().clone(),
                "steps": env.episode_length_buf.detach().clone(),
                "control_step": env.common_step_counter,
            }
            return None, None

    @configclass
    class EvaluationRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        dataset_export_dir_path = "outputs/rl_progress/il_evaluation"
        transition = RecorderTermCfg(class_type=EvaluationRecorder)

    return EvaluationRecorderCfg()


def evaluate_il_policy(args) -> int:
    num_envs = getattr(args, "num_envs", 16)
    n_episodes = getattr(args, "n_episodes", 50)
    seconds = getattr(args, "episode_length_s", 20.0)
    validate_positive_count("num_envs", num_envs)
    validate_positive_count("n_episodes", n_episodes)
    validate_positive_count("control_fps", getattr(args, "control_fps", None))
    if num_envs is None or n_episodes is None:
        raise ValueError("IL 评估需要明确的环境数量和 episode 数量")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("episode_length_s 必须为有限的正数")
    folder = resolve_policy_path(args.policy_path)
    settings = load_simulation_settings(folder, getattr(args, "control_fps", None))
    args.policy_path = str(folder)
    requested_output = getattr(args, "output", None)
    output = Path(requested_output).expanduser().resolve() if requested_output else Path(
        f"outputs/rl_progress/il_eval_{time.time_ns()}.json").resolve()
    if output.exists():
        raise FileExistsError(output)
    sources = {str(path.relative_to(folder)): file_digest(path)
               for path in sorted(folder.rglob("*")) if path.is_file()}
    actual_envs = min(num_envs, n_episodes)

    with ExitStack() as cleanup:
        app = _launch_isaac_app(args, cleanup, enable_cameras=True)
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        import openso101.tasks  # noqa: F401
        from openso101.il.evaluation import CohortEvaluation
        from openso101.il.observations import _build_il_policy_observation_batched
        from openso101.il.policies import load_policy
        from openso101.il.policies.inference import select_sim_action
        from openso101.robots import SO101_SIM_JOINT_NAMES

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=actual_envs,
                            use_fabric=not args.disable_fabric)
        configure_evaluation(cfg, environments=actual_envs, seconds=seconds,
                             scene=getattr(args, "scene", None), settings=settings)
        if getattr(args, "seed", None) is not None:
            cfg.seed = int(args.seed)
        env = gym.make(args.task, cfg=cfg)
        cleanup.callback(env.close)
        runtime = env.unwrapped
        if "time_out" not in runtime.termination_manager.active_terms or runtime.termination_manager.active_terms[-1] != "success":
            raise ValueError("IL 评估需要明确的 timeout，并且 success 必须在全部结束条件之后计算")
        policy = load_policy(folder, device=str(runtime.device))
        if "control_dt" in getattr(policy, "metadata", {}) and abs(policy.metadata["control_dt"] - runtime.step_dt) > 1e-6:
            raise ValueError("student 控制周期与当前环境不一致，请使用 rl student-eval")
        campaign = CohortEvaluation(n_episodes, actual_envs, runtime.device)
        robot = runtime.scene["robot"]
        joint_ids = [list(robot.joint_names).index(name) for name in SO101_SIM_JOINT_NAMES]
        if getattr(args, "seed", None) is not None:
            torch.manual_seed(args.seed)
        policy_resets = 0
        while app.is_running() and not campaign.finished:
            campaign.begin_cohort()
            env.reset()
            if hasattr(policy, "reset"):
                policy.reset()
                policy_resets += 1
            cohort_steps = 0
            while app.is_running() and not campaign.cohort_finished:
                with torch.inference_mode():
                    observation = _build_il_policy_observation_batched(runtime.scene)
                    if getattr(policy, "metadata", {}).get("goal_input") == "robot_root_xyz_m":
                        from openso101.rl.student import student_goal

                        observation["observation.goal"] = student_goal(runtime)
                    action = select_sim_action(policy, observation, runtime.device, actual_envs)
                    if bool((~campaign.active).any()):
                        action[~campaign.active] = robot.data.joint_pos[~campaign.active][:, joint_ids]
                    env.step(action)
                transition = runtime._il_eval_transition
                if transition["control_step"] != runtime.common_step_counter:
                    raise ValueError("IL 评估需要当前物理步骤的结束记录")
                campaign.consume(*(transition[name] for name in ("success", "terminated", "truncated", "steps")))
                cohort_steps += 1
                if not campaign.cohort_finished and cohort_steps >= runtime.max_episode_length:
                    raise RuntimeError("IL 评估 episode 未在配置的控制步骤数量内结束")
        result = {**campaign.report(),
                  "status": "il_evaluation_completed" if campaign.finished else "il_evaluation_interrupted",
                  "task": args.task, "scene": getattr(args, "scene", None), "seed": getattr(args, "seed", None),
                  "requested_environments": num_envs, "actual_environments": actual_envs,
                  "episode_length_s": seconds, "control_dt": float(runtime.step_dt), "policy_resets": policy_resets,
                  "termination_terms": list(runtime.termination_manager.active_terms),
                  "simulation_settings": settings.model_dump(mode="json") if settings is not None else None,
                  "maximum_episode_steps": int(runtime.max_episode_length),
                  "policy_sources": sources, "source_sha256": file_digest(Path(__file__)),
                  "policy_inference_sha256": file_digest(Path(__file__).parents[1] / "policies/inference.py"),
                  "statistics_sha256": file_digest(Path(__file__).parents[1] / "evaluation.py"),
                  "task_success_sha256": file_digest(Path(__file__).parents[2] / "teleop/success.py"),
                  "training_started": False}
    result["native_resources_closed"] = True
    if {str(path.relative_to(folder)): file_digest(path) for path in folder.rglob("*") if path.is_file()} != sources:
        raise ValueError("IL 评估期间模型文件 SHA256 发生变化")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    if not campaign.finished:
        raise RuntimeError(f"IL 评估尚未完成，当前记录已经保存: {output}")
    return 0
