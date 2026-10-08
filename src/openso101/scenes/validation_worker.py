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

    from openso101.rl.gpu_scope import configure_visible_gpu

    configure_visible_gpu()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=args.cameras).app
    import gymnasium as gym
    import h5py
    import numpy as np
    import torch
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils import configclass

    from .runtime import CustomSceneEnvCfg, register_custom_scene, scene_states, scene_jaw_forces
    from .program import TaskProgramTracker
    from .usd import verify_compilation
    from .models import file_digest

    compilation = verify_compilation(args.scene)
    register_custom_scene()
    cfg = CustomSceneEnvCfg()
    cfg.configure_play(True)
    cfg.configure_scene(args.scene)
    cfg.configure_cameras(args.cameras)
    cfg.scene.num_envs = args.num_envs
    trackers = ([TaskProgramTracker(cfg.scene_spec, cfg.task_program) for _ in range(args.num_envs)]
                if cfg.task_program is not None else [])
    bddl_trackers = []
    if cfg.bddl_report is not None:
        from .bddl import BDDLBinding, BDDLTaskTracker

        binding = BDDLBinding.model_validate(cfg.bddl_report["binding"])
        bddl_trackers = [BDDLTaskTracker(cfg.bddl_report["problem"], binding, cfg.scene_spec)
                         for _ in range(args.num_envs)]
    trace = []
    dynamic = [item.entity_id for item in cfg.scene_spec.entities if item.dynamic]

    class ProgramRecorder(RecorderTerm):
        def record_post_reset(self, env_ids):
            for index in env_ids.tolist():
                if trackers:
                    trackers[index].reset()
                if bddl_trackers:
                    bddl_trackers[index].reset()
            return None, None

        def record_post_step(self):
            runtime = self._env
            states = scene_states(runtime).cpu().numpy()
            forces = {name: value.cpu().numpy() for name, value in scene_jaw_forces(runtime).items()}
            jaw_id = runtime.scene["robot"].joint_names.index("Jaw")
            opened = (runtime.scene["robot"].data.joint_pos[:, jaw_id] > .4).cpu().numpy()
            phase = (runtime._scene_program_phase.cpu().numpy().copy() if trackers
                     else np.zeros(args.num_envs, dtype=np.int64))
            phase_hold = (runtime._scene_program_hold.cpu().numpy().copy() if trackers
                          else np.zeros(args.num_envs))
            held = runtime._scene_hold_seconds.cpu().numpy().copy()
            success = runtime._scene_success.cpu().numpy().copy()
            bddl_eligible = []
            for index, tracker in enumerate(bddl_trackers):
                measured = {item.entity_id: states[index, column]
                            for column, item in enumerate(cfg.scene_spec.entities)}
                result = tracker.update(measured, bool(opened[index]), runtime.step_dt)
                eligible = result["source_goal_satisfied"] and result["stable"]
                if cfg.scene_spec.task.require_released:
                    eligible = eligible and bool(opened[index]) and all(
                        (forces[name][index] <= .1).all() for name in tracker.subject_ids)
                bddl_eligible.append(bool(eligible))
                if not trackers:
                    tracker.elapsed = tracker.elapsed if eligible else 0.
                    expected = eligible and tracker.elapsed + 1e-7 >= cfg.scene_spec.task.settle_seconds
                    if expected != success[index] or abs(tracker.elapsed - held[index]) > 1e-5:
                        raise RuntimeError(f"BDDL 的实际状态检查失败：environment={index}")
            for index, tracker in enumerate(trackers):
                values = {item.entity_id: states[index, column]
                          for column, item in enumerate(cfg.scene_spec.entities)}
                contacts = {name: value[index] for name, value in forces.items()}
                result = tracker.update(values, contacts, bool(opened[index]), runtime.step_dt,
                                        additional_eligible=bddl_eligible[index] if bddl_trackers else True)
                if (result["phase"] != phase[index] or result["success"] != success[index]
                        or abs(tracker.phase_hold_seconds - phase_hold[index]) > 1e-5
                        or abs(result["held_seconds"] - held[index]) > 1e-5):
                    raise RuntimeError(f"TaskProgram 的实际状态检查失败：environment={index}")
            trace.append({"states": states, "jaw_forces": np.stack([forces[name] for name in dynamic], axis=1),
                          "opened": opened, "phase": phase, "phase_hold": phase_hold,
                          "final_hold": held, "success": success,
                          "terminated": runtime.reset_terminated.cpu().numpy().copy(),
                          "truncated": runtime.reset_time_outs.cpu().numpy().copy()})
            return None, None

    @configclass
    class ProgramRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_dir_path = str(args.output.parent / "recorder")
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        program = RecorderTermCfg(class_type=ProgramRecorder)

    cfg.recorders = ProgramRecorderCfg()
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
        if len(trace) != args.steps:
            raise RuntimeError("场景实际运行状态记录数量不完整")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trace_path = args.output.with_suffix(".hdf5")
        with h5py.File(trace_path, "x") as stream:
            for name in trace[0]:
                stream.create_dataset(name, data=np.stack([item[name] for item in trace]))
            stream.attrs["entity_ids"] = json.dumps([item.entity_id for item in cfg.scene_spec.entities])
            stream.attrs["dynamic_entity_ids"] = json.dumps(dynamic)
            stream.attrs["scene_sha256"] = compilation["scene_sha256"]
            stream.attrs["control_dt"] = env.unwrapped.step_dt
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
            "trace_sha256": file_digest(trace_path),
            "program_conditions_verified": bool(trackers),
            "bddl_conditions_verified": bool(bddl_trackers),
            "bddl_checked_frames": args.steps * args.num_envs if bddl_trackers else 0,
            "bddl_initial_conditions_verified": False,
            "program_checked_frames": args.steps * args.num_envs if trackers else 0,
            "program_maximum_phase": int(max(item["phase"].max() for item in trace)),
            "observed_dual_contact_frames": int(sum((item["jaw_forces"] > .5).all(axis=-1).sum() for item in trace)),
            "pending_checks": ["dynamic_stability", "path_reachability", "contact_geometry", "camera_visibility", "task_completion", "successful_collection"],
        }
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
    finally:
        env.close()
    app.close()


if __name__ == "__main__":
    main()
