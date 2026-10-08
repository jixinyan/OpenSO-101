import argparse
import json
from pathlib import Path

from openso101.rl.gpu_scope import configure_visible_gpu
from openso101.scenes.models import file_digest


def maximum_state_error(saved, current):
    errors = {}
    for name, value in saved.items():
        if isinstance(value, dict):
            errors.update({f"{name}/{key}": error for key, error in maximum_state_error(value, current[name]).items()})
        else:
            if value.shape != current[name].shape:
                raise ValueError(f"原生 checkpoint 状态尺寸不一致: {name}")
            errors[name] = float((value.double() - current[name].double()).abs().max())
    if errors and max(errors.values()) > 0.000001:
        raise ValueError(f"原生 checkpoint 恢复误差超过 1e-6: {errors}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0", "OpenSO101-Stack-v0", "OpenSO101-CustomScene-v0"), required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.task == "OpenSO101-CustomScene-v0") != (args.scene is not None):
        raise ValueError("CustomScene checkpoint 检查需要对应 scene bundle")
    if args.scene is not None and not args.scene.is_dir():
        raise NotADirectoryError(args.scene)
    physical_gpu = configure_visible_gpu()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    env, recorder = None, None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import openso101.tasks  # noqa: F401
        from openso101.robots import SO101_SIM_JOINT_NAMES
        from openso101.teleop.checkpoints import _TeleopCheckpointStore
        from openso101.teleop.recorder.hdf5 import OpenSO101HDF5TeleopRecorder, validate_hdf5_episode
        from openso101.teleop.recorder.lerobot import collect_camera_buffers, discover_camera_metadata, read_robot_proprio
        from openso101.teleop.timing import _env_control_rate_fps

        if args.scene is not None:
            from openso101.scenes.isaaclab.runtime import register_custom_scene

            register_custom_scene()
        cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
        cfg.configure_play(True)
        cfg.configure_action_mode("teleop")
        cfg.configure_cameras(True)
        cfg.seed = 1000
        if args.scene is not None:
            cfg.configure_scene(args.scene)
        env = gym.make(args.task, cfg=cfg)
        env.reset()
        runtime = env.unwrapped
        scene = runtime.scene
        robot = scene["robot"]
        ids = [list(robot.joint_names).index(name) for name in SO101_SIM_JOINT_NAMES]
        held = robot.data.joint_pos[:, ids].clone()
        recorder = OpenSO101HDF5TeleopRecorder(args.output / "dataset", args.task,
                                             discover_camera_metadata(scene), _env_control_rate_fps(runtime),
                                             sim_joint_names=SO101_SIM_JOINT_NAMES, env_id=args.task, flush_steps=4)
        recorder.start_episode()

        def record_step(action):
            _, _, terminated, truncated, _ = env.step(action)
            if bool((terminated | truncated).any()):
                raise RuntimeError("原生 checkpoint 检查期间 episode 已经终止")
            qpos, qvel = read_robot_proprio(robot)
            recorder.add_frame(action=action[0].detach().cpu().numpy(), qpos=qpos, qvel=qvel,
                               timestamp=float(runtime.episode_length_buf[0]) * runtime.step_dt,
                               camera_buffers=collect_camera_buffers(scene))

        with torch.inference_mode():
            for _ in range(4):
                record_step(held)
            store = _TeleopCheckpointStore(runtime, scene, SO101_SIM_JOINT_NAMES)
            store.capture(recorder)
            saved = store.checkpoint
            moved = held.clone()
            moved[:, :5] += 0.03
            for _ in range(6):
                record_step(moved)
            hold_target = store.restore(recorder)
            runtime.sim.forward()
            scene.update(0.0)
            errors = maximum_state_error(saved.scene_state, scene.get_state(is_relative=False))
            errors.update({f"collection/{key}": value for key, value in maximum_state_error(
                saved.collection_states, {name: entity.data.object_state_w for name, entity in scene.rigid_object_collections.items()}).items()})
            errors.update({f"env/{key}": value for key, value in maximum_state_error(
                saved.env_state, {key: getattr(runtime, key) for key in saved.env_state}).items()})
            errors.update({f"action/{key}": value for key, value in maximum_state_error(
                saved.action_state, {key: getattr(runtime.action_manager, key) for key in saved.action_state}).items()})
            for name, state in saved.command_states.items():
                command = runtime.command_manager.get_term(name)
                errors.update({f"command/{name}/{key}": value for key, value in maximum_state_error(
                    state, {key: getattr(command, key) for key in state}).items()})
            success = store._success()
            errors.update({f"success/{key}": value for key, value in maximum_state_error(
                saved.success_state, {key: getattr(success, key) for key in saved.success_state}).items()})
            if hold_target.shape != (6,) or not torch.equal(hold_target, saved.hold_joint_target) or recorder.total_frames != 4:
                raise RuntimeError("原生 checkpoint 的保持姿态或记录裁剪不一致")
            if tuple(tracker.snapshot() for tracker in store._trackers()) != saved.bddl_progress:
                raise RuntimeError("原生 checkpoint 的 BDDL 状态恢复不一致")
            for _ in range(4):
                record_step(hold_target.unsqueeze(0))
        saved_frames = recorder.total_frames
        episode = recorder.save_episode(success=False)
        validate_hdf5_episode(episode)
        result = {"status": "native_teleop_checkpoint_verified", "task": args.task,
                  "scene": str(args.scene) if args.scene is not None else None, "physical_gpu": physical_gpu,
                  "restored_state_maximum_errors": errors, "saved_frames": saved_frames,
                  "episode": str(episode), "episode_sha256": file_digest(episode),
                  "checkpoint_source_sha256": file_digest(Path("src/openso101/teleop/checkpoints.py")),
                  "validator_source_sha256": file_digest(Path(__file__)),
                  "native_restore_verified": True, "task_success_verified": False, "training_started": False}
    finally:
        try:
            if recorder is not None:
                recorder.cancel_episode()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()
    result["native_resources_closed"] = True
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
