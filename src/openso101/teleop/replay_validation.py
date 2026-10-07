from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import h5py


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReplayValidation:
    def __init__(self, episode: Path, report_path: Path, task: str, h5, env, frame_range: range, checkpoint_frame: int):
        self.report_path = report_path
        self.trajectory = []
        if report_path.with_suffix(".hdf5").exists():
            raise FileExistsError(report_path.with_suffix(".hdf5"))
        source_fps = int(h5.attrs.get("fps", 30))
        timing_error = abs(float(env.step_dt) - 1.0 / source_fps)
        if timing_error > 1e-6:
            raise ValueError(f"回放控制周期与记录 FPS 不匹配: {env.step_dt}, {source_fps}")
        self.camera_shapes = {
            name: tuple(h5[f"observations/images/{name}"].shape[1:])
            for name in ("wrist_camera", "overhead_camera")
        }
        self.report = {
            "status": "replay_running",
            "task": task,
            "source_episode": str(episode),
            "source_sha256": _file_sha256(episode),
            "source_task_success": bool(h5.attrs.get("success", False)),
            "source_scene_sha256": str(h5.attrs.get("scene_sha256", "")),
            "source_sim_fields": sorted(h5["sim"].keys()) if "sim" in h5 else [],
            "observed_env_index": 0,
            "checkpoint_frame": checkpoint_frame,
            "start_frame": frame_range.start,
            "stop_frame": frame_range.stop,
            "requested_frames": len(frame_range),
            "completed_frames": 0,
            "control_dt": float(env.step_dt),
            "source_fps": source_fps,
            "control_timing_error_seconds": timing_error,
            "step_checks": {"warm_start": 0, "hold": 0, "replay": 0},
            "maximum_action_error": 0.0,
            "source_num_envs": int(h5.attrs.get("source_num_envs", 1)),
            "replay_num_envs": env.num_envs,
            "source_cohort_restored": bool(getattr(env, "_replay_cohort", False)),
            "maximum_cohort_action_error": 0.0,
            "restore_errors": {},
            "camera_checks": {name: {"frames": 0, "minimum_pixel_std": None} for name in self.camera_shapes},
            "task_success_verified": False,
            "native_task_configuration": hasattr(env, "_replay_mapping"),
            "physics_state_reproducibility_verified": False,
            "validator_sha256": _file_sha256(Path(__file__)),
            "cli_sha256": _file_sha256(Path(__file__).parents[1] / "cli" / "il.py"),
        }

    def check_restore(self, env, h5, frame_index: int) -> None:
        from openso101.cli.il import _collect_replay_sim_state, _replay_robot_joint_indices

        robot = env.scene["robot"]
        ids = _replay_robot_joint_indices(robot)
        pairs = {
            "joint_position": (robot.data.joint_pos[0, ids], h5["observations/qpos"][frame_index]),
            "joint_velocity": (robot.data.joint_vel[0, ids], h5["observations/qvel"][frame_index]),
        }
        actual_state = _collect_replay_sim_state(env, env.scene, include_cohort=self.report["source_cohort_restored"])
        origin_delta = (actual_state["environment_origin"] - h5["sim/environment_origin"][frame_index]
                        if "sim/environment_origin" in h5 else np.zeros(3))
        self.report["replay_environment_origin"] = actual_state["environment_origin"].tolist()
        if "sim/environment_origin" in h5:
            self.report["source_environment_origin"] = h5["sim/environment_origin"][frame_index].tolist()
        for field in self.report["source_sim_fields"]:
            if field == "environment_origin":
                continue
            if field in ("cohort_joint_targets", "cohort_policy_actions") or (
                    field.startswith("cohort_") and not self.report["source_cohort_restored"]):
                continue
            expected = h5[f"sim/{field}"][frame_index].copy()
            if field in ("object_root_state", "command_goal_pos_w"):
                expected[:3] += origin_delta
            pairs[field] = (actual_state[field], expected)
        for field, (actual, expected) in pairs.items():
            actual = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError(f"恢复状态格式错误: {field}")
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            if error > 1e-6:
                raise ValueError(f"恢复状态误差超过 1e-6: {field}={error}")
            self.report["restore_errors"][field] = error

    def check_step(self, env, phase: str, expected_action) -> None:
        expected_action = np.asarray(expected_action)
        if self.report["source_cohort_restored"]:
            actual = env._replay_transition["cohort_targets"].detach().cpu().numpy()
            if actual.shape != expected_action.shape or not np.isfinite(actual).all():
                raise ValueError("并行回放动作格式错误")
            error = float(np.max(np.abs(actual.astype(np.float64) - expected_action.astype(np.float64))))
            if error > 1e-6:
                raise ValueError(f"并行回放动作误差超过 1e-6: {error}")
            self.report["maximum_cohort_action_error"] = max(self.report["maximum_cohort_action_error"], error)
            expected_action = expected_action[0]
        if hasattr(env, "_replay_mapping"):
            action = env._replay_transition["targets"].detach().cpu().numpy()
            self.report["task_success_verified"] |= env._replay_transition["success"]
            self.trajectory.append({name: value.detach().cpu().numpy().copy()
                                    for name, value in env._replay_transition.items() if isinstance(value, torch.Tensor)})
        else:
            action = env.action_manager.action[0].detach().cpu().numpy()
        expected_action = np.asarray(expected_action)
        if action.shape != expected_action.shape or not np.isfinite(action).all():
            raise ValueError("回放动作格式错误")
        error = float(np.max(np.abs(action.astype(np.float64) - expected_action.astype(np.float64))))
        if error > 1e-6:
            raise ValueError(f"回放动作误差超过 1e-6: {error}")
        self.report["maximum_action_error"] = max(self.report["maximum_action_error"], error)
        robot = env.scene["robot"]
        if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(robot.data.joint_vel).all():
            raise ValueError("回放关节状态包含非有限值")
        for name, expected_shape in self.camera_shapes.items():
            rgb = env.scene[name].data.output["rgb"][0, :, :, :3]
            if tuple(rgb.shape) != expected_shape or not torch.isfinite(rgb).all():
                raise ValueError(f"回放相机数据格式错误: {name}")
            camera_report = self.report["camera_checks"][name]
            camera_report["frames"] += 1
            pixel_std = float(rgb.float().std())
            previous = camera_report["minimum_pixel_std"]
            camera_report["minimum_pixel_std"] = pixel_std if previous is None else min(previous, pixel_std)
        self.report["step_checks"][phase] += 1
        if phase == "replay":
            self.report["completed_frames"] += 1

    def save(self) -> None:
        if self.trajectory:
            path = self.report_path.with_suffix(".hdf5")
            with h5py.File(path, "x") as stream:
                for name in self.trajectory[0]:
                    stream.create_dataset(name, data=np.stack([item[name] for item in self.trajectory]))
            self.report["trajectory_sha256"] = _file_sha256(path)
            self.report["maximum_hold_seconds"] = max(float(item["hold_seconds"]) for item in self.trajectory)
        self.report["status"] = (
            "replay_verified" if self.report["completed_frames"] == self.report["requested_frames"]
            else "replay_interrupted"
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(self.report, indent=2), encoding="utf-8")


def native_replay_recorder():
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils import configclass

    class NativeReplayRecorder(RecorderTerm):
        def record_post_step(self):
            runtime = self._env
            from openso101.robots import SO101_SIM_JOINT_NAMES
            from openso101.tasks.shared.grasp import _jaw_force_magnitude
            from isaaclab.utils.math import subtract_frame_transforms

            robot = runtime.scene["robot"]
            ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
            hold = (runtime.command_manager.get_term("object_pose").placement_hold_seconds
                    if runtime.cfg.task_profile_task == "pick_place"
                    else runtime.termination_manager.get_term_cfg("success").func.hold_seconds)
            obj = runtime.scene["object"]
            object_position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                            obj.data.root_pos_w, obj.data.root_quat_w)
            runtime._replay_transition = {
                "targets": torch.cat([runtime.action_manager.get_term(name).processed_actions[0]
                                      for name in runtime.action_manager.active_terms]).detach().clone(),
                "success": bool(runtime.termination_manager.get_term("success")[0]),
                "joint_position": robot.data.joint_pos[0, ids].detach().clone(),
                "joint_velocity": robot.data.joint_vel[0, ids].detach().clone(),
                "object_root_state": runtime.scene["object"].data.root_state_w[0].detach().clone(),
                "object_position_root": object_position[0].detach().clone(),
                "jaw_forces": torch.stack([_jaw_force_magnitude(runtime.scene[name])[0]
                                            for name in ("gripper_jaw_contact", "moving_jaw_contact")]).detach().clone(),
                "hold_seconds": hold[0].detach().clone(),
            }
            if getattr(runtime, "_replay_cohort", False):
                runtime._replay_transition["cohort_targets"] = torch.cat([
                    runtime.action_manager.get_term(name).processed_actions
                    for name in runtime.action_manager.active_terms], dim=-1).detach().clone()
            return None, None

    @configclass
    class NativeReplayRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        dataset_export_dir_path = "outputs/rl_progress/replay_manager"
        transition = RecorderTermCfg(class_type=NativeReplayRecorder)

    return NativeReplayRecorderCfg()
