from pathlib import Path

import numpy as np
import torch
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils import configclass

from openso101.cli.il import _collect_replay_sim_state
from openso101.robots import SO101_SIM_JOINT_NAMES
from openso101.teleop.hdf5_recorder import OpenSO101HDF5TeleopRecorder, validate_hdf5_episode
from openso101.teleop.lerobot_recorder import collect_camera_buffers


def first_episode_recorder(output: Path, task_id: str, task_profile: str, policy_sha256: str):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    class PolicyRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.recording = None
            self.finished = False

        def record_pre_step(self):
            if self.finished:
                return None, None
            runtime = self._env
            robot = runtime.scene["robot"]
            images = collect_camera_buffers(runtime.scene)
            if self.recording is None:
                fps = round(1 / runtime.step_dt)
                if not np.isclose(1 / fps, runtime.step_dt, atol=1e-9, rtol=0):
                    raise ValueError("HDF5 采集需要整数控制频率")
                cameras = {name: {"height": image.shape[0], "width": image.shape[1]} for name, image in images.items()}
                scene_metadata = ({"scene_sha256": runtime.cfg.scene_spec.digest()}
                                  if hasattr(runtime.cfg, "scene_spec") else None)
                self.recording = OpenSO101HDF5TeleopRecorder(
                    output, task_name=task_id, env_id=task_id, cameras=cameras, fps=fps,
                    dataset_id="local/openso101_policy_evaluation", scene_metadata=scene_metadata,
                    sim_joint_names=SO101_SIM_JOINT_NAMES)
                self.recording.start_episode()
                self.recording._h5.attrs["policy_sha256"] = policy_sha256
                self.recording._h5.attrs["time_base"] = "simulation"
                self.recording._h5.attrs["controller"] = "actual_policy_joint_targets"
                self.recording._h5.attrs["task_profile"] = task_profile
            ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
            targets = torch.cat([runtime.action_manager.get_term(name).processed_actions
                                 for name in runtime.action_manager.active_terms], dim=-1)
            self.recording.add_frame(
                action=targets[0].cpu().numpy(), qpos=robot.data.joint_pos[0, ids].cpu().numpy(),
                qvel=robot.data.joint_vel[0, ids].cpu().numpy(), camera_buffers=images,
                timestamp=runtime.common_step_counter * runtime.step_dt,
                sim_state=_collect_replay_sim_state(runtime, runtime.scene))
            return None, None

        def record_post_step(self):
            runtime = self._env
            if not self.finished and (runtime.reset_terminated[0] or runtime.reset_time_outs[0]):
                success = any(bool(runtime.termination_manager.get_term(name)[0]) for name in
                              runtime.termination_manager.active_terms if "success" in name)
                episode = self.recording.save_episode(success=success)
                validate_hdf5_episode(episode)
                self.finished = True
            return None, None

    @configclass
    class PolicyRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        dataset_export_dir_path = str(output / "manager")
        policy_recording = RecorderTermCfg(class_type=PolicyRecorder)

    return PolicyRecorderCfg()
