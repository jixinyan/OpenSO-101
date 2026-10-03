import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--environment-mode", choices=("nominal", "randomized"), default="nominal")
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()
if args.output.exists() or args.steps <= 0:
    raise ValueError("检查需要新的输出目录和正数 steps")
args.task_profile = "grasp_v3"
args.num_envs = 4
args.seed = 42
args.with_cameras = False
args.visual_dr = False

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    import h5py
    import torch
    from tensordict import TensorDict
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils import configclass

    from openso101.rl.config import digest
    from openso101.rl.execution import build_environment
    from openso101.rl.bounded_policy import BoundedActorCritic
    from openso101.rl.portable import decode_joint_targets
    from openso101.rl.vision_distillation import action_mapping
    from openso101.robots import SO101_SIM_JOINT_NAMES
    from openso101.tasks.shared.grasp_profile import success_event
    from openso101.tasks.shared import grasp_v3

    physics_samples = []
    physics_positions = []
    target_checks = []

    class PhysicsSpeedRecorder(RecorderTerm):
        def record_pre_step(self):
            runtime = self._env
            robot = runtime.scene["robot"]
            ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
            before = robot.data.joint_pos[:, ids].clone()
            actual = torch.cat([runtime.action_manager.get_term(name).processed_actions
                                for name in runtime.action_manager.active_terms], dim=-1)
            applied = runtime.action_manager.action
            decoded = decode_joint_targets(applied, mapping, joint_position=before)
            error = float((decoded - actual).abs().max())
            unconstrained = before + applied.clamp(-1, 1) * (2 * runtime.step_dt)
            correction = float((actual - unconstrained).abs().max())
            if error > 1e-6:
                raise RuntimeError(f"关节目标检查失败：error={error}")
            target_checks.append((error, correction, before.cpu(), applied.cpu().clone()))
            return None, None

        def record_post_physics_decimation_step(self):
            asset = self._env.scene["robot"]
            joint_ids = [asset.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
            physics_samples.append(asset.root_physx_view.get_dof_velocities()[:, joint_ids].clone())
            physics_positions.append(asset.root_physx_view.get_dof_positions()[:, joint_ids].clone())
            return None, None

    @configclass
    class RuntimeRecorderCfg(RecorderManagerBaseCfg):
        dataset_export_dir_path = str(args.output.parent / "recorder")
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        speed = RecorderTermCfg(class_type=PhysicsSpeedRecorder)

    args.recorder_cfg = RuntimeRecorderCfg()
    env = build_environment(args, training=True)
    observation, _ = env.reset()
    runtime = env.unwrapped
    robot = runtime.scene["robot"]
    ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
    mapping = action_mapping(runtime)
    torch.manual_seed(args.seed)
    actor = BoundedActorCritic(TensorDict(observation, batch_size=[4]),
                             {"policy": ["policy"], "critic": ["policy"]}, 6,
                             noise_std_type="log", init_noise_std=.8).to(runtime.device)
    maximum_target_error = 0.
    maximum_speed = 0.
    maximum_limit_correction = 0.
    resets = 0
    shaping_returns = torch.zeros(4, device=runtime.device, dtype=torch.float64)
    episode_steps = torch.zeros(4, device=runtime.device, dtype=torch.int64)
    completed_shaping_returns = []
    activity_returns = torch.zeros_like(shaping_returns)
    completed_activity_returns = []
    records = []
    for index in range(args.steps):
        with torch.no_grad():
            actions = actor.act(TensorDict(observation, batch_size=[4]))
            log_probability = actor.get_actions_log_prob(actions)
            entropy = actor.entropy
            if (not torch.isfinite(log_probability).all() or not torch.isfinite(entropy).all()
                    or (actions.abs() >= 1).any()):
                raise RuntimeError("bounded actor 的分布检查失败")
            observation, reward, terminated, truncated, _ = env.step(actions)
            if len(target_checks) != index + 1:
                raise RuntimeError("控制步骤的动作记录数量不一致")
            error, correction, before, applied_actions = target_checks[index]
            maximum_target_error = max(maximum_target_error, error)
            maximum_limit_correction = max(maximum_limit_correction, correction)
            progress_index = runtime.reward_manager.active_terms.index("progress")
            shaping = runtime.reward_manager._step_reward[:, progress_index] * runtime.step_dt
            shaping_returns += runtime.cfg.reward_discount ** episode_steps * shaping.double()
            if "task_activity" in runtime.reward_manager.active_terms:
                activity_index = runtime.reward_manager.active_terms.index("task_activity")
                activity_returns += runtime.reward_manager._step_reward[:, activity_index].double() * runtime.step_dt
            episode_steps += 1
            done = terminated | truncated
            completed_shaping_returns.extend(shaping_returns[done].cpu().tolist())
            completed_activity_returns.extend(activity_returns[done].cpu().tolist())
            shaping_returns[done] = 0
            activity_returns[done] = 0
            episode_steps[done] = 0
            speed = float(robot.data.joint_vel[:, ids].abs().max())
            maximum_speed = max(maximum_speed, speed)
            if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("真实环境产生无效观测或 reward")
            event = success_event(runtime)
            if (event.bool() & ~terminated).any():
                raise RuntimeError("成功奖励必须对应当前结束步骤")
            resets += int((terminated | truncated).sum())
            records.append((actions.cpu(), before.cpu(), reward.cpu(), event.cpu()))
    args.output.mkdir(parents=True)
    physics_velocity = torch.stack(physics_samples).cpu()
    maximum_physics_speed = float(physics_velocity.abs().max())
    if len(physics_samples) != args.steps * runtime.cfg.decimation or not torch.isfinite(physics_velocity).all():
        raise RuntimeError("物理步骤的实际速度记录不完整或含有无效数值")
    if completed_shaping_returns and max(abs(value) for value in completed_shaping_returns) > 1e-4:
        raise RuntimeError("完整 episode 的 shaping 累计检查失败")
    if completed_activity_returns and (min(completed_activity_returns) < 0
                                       or max(completed_activity_returns) > .25 * runtime.cfg.episode_length_s + 1e-5):
        raise RuntimeError("完整 episode 的 task_activity 累计超出配置范围")
    with h5py.File(args.output / "runtime.hdf5", "w") as trace:
        for column, name in enumerate(("raw_action", "joint_position_before", "reward", "success_event")):
            trace.create_dataset(name, data=torch.stack([record[column] for record in records]).numpy())
        trace.create_dataset("physics_steps/joint_velocity", data=physics_velocity.numpy())
        trace.create_dataset("physics_steps/joint_position", data=torch.stack(physics_positions).cpu().numpy())
        trace.create_dataset("applied_action", data=torch.stack([record[3] for record in target_checks]).numpy())
    report = {"task": args.task, "task_profile": args.task_profile, "environment_mode": args.environment_mode,
              "steps": args.steps, "num_envs": 4, "resets": resets,
              "seed": args.seed,
              "observation_dim": observation["policy"].shape[-1], "maximum_target_error_rad": maximum_target_error,
              "maximum_sampled_speed_rad_s": maximum_speed,
              "maximum_physics_speed_rad_s": maximum_physics_speed,
              "actual_velocity_limit_verified": maximum_physics_speed <= 2.001,
              "physics_dt": runtime.physics_dt, "control_dt": runtime.step_dt,
              "maximum_target_limit_correction_rad": maximum_limit_correction,
              "distribution_and_target_mapping_verified": True, "task_success_verified": False,
              "completed_episode_shaping_returns": completed_shaping_returns,
              "completed_episode_activity_returns": completed_activity_returns,
              "trace_sha256": digest(args.output / "runtime.hdf5"), "source_sha256": digest(Path(__file__)),
              "profile_sha256": digest(Path(grasp_v3.__file__))}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
finally:
    if env is not None:
        env.close()
app.close()
