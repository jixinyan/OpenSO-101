from dataclasses import dataclass, field
from typing import Any

from .sim_state import _REPLAY_COMMAND_FIELDS


_ENV_FIELDS = ("episode_length_buf", "_scene_hold_seconds", "_scene_success", "_scene_program_phase", "_scene_program_hold")
_COMMAND_FIELDS = (*_REPLAY_COMMAND_FIELDS, "time_left", "command_counter", "just_completed_stage")


def _snapshot_fields(owner, names):
    import torch

    result = {}
    for name in names:
        if hasattr(owner, name):
            value = getattr(owner, name)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"checkpoint 的 {name} 需要包含有限数值的 Tensor")
            result[name] = value.clone()
    return result


def _validate_fields(owner, saved):
    for name, value in saved.items():
        current = getattr(owner, name)
        if current.shape != value.shape or current.dtype != value.dtype or current.device != value.device:
            raise ValueError(f"checkpoint 的 {name} 与当前环境不一致")


def _restore_fields(owner, saved):
    _validate_fields(owner, saved)
    for name, value in saved.items():
        getattr(owner, name).copy_(value)


def _validate_scene_state(saved, current):
    import torch

    if saved.keys() != current.keys():
        raise ValueError("checkpoint 与当前场景的实体或状态字段不一致")
    for name, value in saved.items():
        target = current[name]
        if isinstance(value, dict):
            _validate_scene_state(value, target)
        elif (not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
              or value.shape != target.shape or value.dtype != target.dtype or value.device != target.device):
            raise ValueError(f"checkpoint 场景状态不一致: {name}")


def _checkpoint_joint_target(robot, joint_pos, sim_joint_names):
    import torch

    names = tuple(sim_joint_names)
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError("checkpoint 需要六个互不重复的 SO-101 关节名称")
    indices = [list(robot.joint_names).index(name) for name in names]
    selected = joint_pos[:, indices]
    if selected.shape != (1, 6) or not torch.isfinite(selected).all():
        raise ValueError("checkpoint 的遥操姿态需要一个环境的六个有限关节值")
    return selected[0].clone()


@dataclass
class _TeleopSimCheckpoint:
    recorder_checkpoint: Any
    scene_state: dict | None = None
    collection_states: dict = field(default_factory=dict)
    hold_joint_target: Any | None = None
    env_state: dict = field(default_factory=dict)
    action_state: dict = field(default_factory=dict)
    command_states: dict = field(default_factory=dict)
    success_state: dict = field(default_factory=dict)
    bddl_progress: tuple[Any, ...] = ()


class _TeleopCheckpointStore:
    def __init__(self, env=None, scene=None, sim_joint_names=()):
        if (env is None) != (scene is None):
            raise ValueError("仿真 checkpoint 需要同时提供 env 和 scene")
        self.env = env
        self.scene = scene
        self.sim_joint_names = tuple(sim_joint_names)
        self.checkpoint: _TeleopSimCheckpoint | None = None

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint is not None

    def _trackers(self):
        if self.env is None or getattr(self.env.cfg, "scene_spec", None) is None:
            return ()
        from openso101.scenes.isaaclab.runtime import bddl_trackers

        return bddl_trackers(self.env)

    def _success(self):
        if "success" in self.env.termination_manager.active_terms:
            return self.env.termination_manager.get_term_cfg("success").func
        return None

    def capture(self, recorder) -> None:
        checkpoint = _TeleopSimCheckpoint(recorder_checkpoint=None)
        if self.env is not None:
            if self.env.num_envs != 1:
                raise ValueError("人工遥操 checkpoint 需要一个环境")
            checkpoint.scene_state = self.scene.get_state(is_relative=False)
            _validate_scene_state(checkpoint.scene_state, checkpoint.scene_state)
            checkpoint.collection_states = {name: entity.data.object_state_w.clone()
                                            for name, entity in self.scene.rigid_object_collections.items()}
            _validate_scene_state(checkpoint.collection_states, checkpoint.collection_states)
            checkpoint.hold_joint_target = _checkpoint_joint_target(
                self.scene["robot"], checkpoint.scene_state["articulation"]["robot"]["joint_position"], self.sim_joint_names)
            checkpoint.env_state = _snapshot_fields(self.env, _ENV_FIELDS)
            checkpoint.action_state = _snapshot_fields(self.env.action_manager, ("action", "prev_action"))
            checkpoint.command_states = {name: _snapshot_fields(self.env.command_manager.get_term(name), _COMMAND_FIELDS)
                                         for name in self.env.command_manager.active_terms}
            checkpoint.success_state = _snapshot_fields(self._success(), ("hold_seconds",))
            checkpoint.bddl_progress = tuple(tracker.snapshot() for tracker in self._trackers())
        checkpoint.recorder_checkpoint = recorder.create_checkpoint()
        self.checkpoint = checkpoint
        print("[INFO]: 已保存遥操 checkpoint。")

    def restore(self, recorder):
        if self.checkpoint is None:
            raise RuntimeError("恢复遥操需要已保存的 checkpoint")
        checkpoint = self.checkpoint
        if self.env is not None:
            _validate_scene_state(checkpoint.scene_state, self.scene.get_state(is_relative=False))
            current_collections = {name: entity.data.object_state_w for name, entity in self.scene.rigid_object_collections.items()}
            _validate_scene_state(checkpoint.collection_states, current_collections)
            _validate_fields(self.env, checkpoint.env_state)
            _validate_fields(self.env.action_manager, checkpoint.action_state)
            if set(checkpoint.command_states) != set(self.env.command_manager.active_terms):
                raise ValueError("checkpoint 的任务命令名称与当前环境不一致")
            for name, state in checkpoint.command_states.items():
                _validate_fields(self.env.command_manager.get_term(name), state)
            _validate_fields(self._success(), checkpoint.success_state)
            if len(self._trackers()) != len(checkpoint.bddl_progress):
                raise ValueError("checkpoint 的 BDDL 环境数量不一致")
        recorder.restore_checkpoint(checkpoint.recorder_checkpoint)
        if self.env is not None:
            import torch

            self.scene.reset_to(checkpoint.scene_state, is_relative=False)
            for name, state in checkpoint.collection_states.items():
                self.scene.rigid_object_collections[name].write_object_state_to_sim(state)
            robot = self.scene["robot"]
            robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
            _restore_fields(self.env, checkpoint.env_state)
            _restore_fields(self.env.action_manager, checkpoint.action_state)
            for name, state in checkpoint.command_states.items():
                _restore_fields(self.env.command_manager.get_term(name), state)
            _restore_fields(self._success(), checkpoint.success_state)
            for tracker, progress in zip(self._trackers(), checkpoint.bddl_progress, strict=True):
                tracker.restore(progress)
            if hasattr(self.env, "_scene_success_step"):
                self.env._scene_success_step = -1
            self.scene.write_data_to_sim()
        print("[INFO]: 已恢复遥操 checkpoint。")
        return checkpoint.hold_joint_target.clone() if checkpoint.hold_joint_target is not None else None
