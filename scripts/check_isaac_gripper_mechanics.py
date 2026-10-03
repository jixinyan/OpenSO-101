import argparse
import json
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--physics-dt", type=float, default=.01)
parser.add_argument("--velocity-iterations", type=int, default=1)
parser.add_argument("--max-depenetration-velocity", type=float, default=5.)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
targets = np.asarray(plan["joint_targets"])
if (args.output.exists() or args.num_envs <= 0 or targets.ndim != 2 or targets.shape[1] != 6
        or not np.isfinite(targets).all() or len(plan["phases"]) != len(targets)
        or not np.isfinite(args.physics_dt) or args.physics_dt <= 0 or args.velocity_iterations <= 0
        or not np.isfinite(args.max_depenetration_velocity) or args.max_depenetration_velocity <= 0):
    raise ValueError("夹爪检查需要有效计划、环境数量与尚未存在的输出目录")
args.task = "OpenSO101-Lift-v0"
args.task_profile = "grasp_v2"
args.seed = 42
args.with_cameras = False

from openso101.rl.gpu_scope import configure_visible_gpu

configure_visible_gpu()
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

    from openso101.rl.config import digest
    import openso101.tasks
    from openso101.tasks.shared.grasp_profile import configure_grasp_profile
    from openso101.robots import SO101_SIM_JOINT_NAMES
    from openso101.tasks.shared.grasp import _jaw_force_magnitude

    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    configure_grasp_profile(cfg, args.task)
    cfg.configure_play(True)
    cfg.scene.num_envs = args.num_envs
    cfg.configure_cameras(False)
    cfg.seed = args.seed
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(plan["control_dt"] / cfg.sim.dt)
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
    for asset in (cfg.scene.robot, cfg.scene.object):
        asset.spawn.rigid_props.max_depenetration_velocity = args.max_depenetration_velocity
    cfg.scene.object.spawn.rigid_props.solver_velocity_iteration_count = args.velocity_iterations
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    runtime = env.unwrapped
    robot = runtime.scene["robot"]
    obj = runtime.scene["object"]
    ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
    substeps = round(plan["control_dt"] / runtime.physics_dt)
    if substeps < 1 or not np.isclose(substeps * runtime.physics_dt, plan["control_dt"]):
        raise ValueError("原生物理周期无法表示夹爪检查周期")
    positions = torch.tensor(plan["initial_joint_position"], device=runtime.device).repeat(args.num_envs, 1)
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), joint_ids=ids)
    local_position = torch.tensor(plan["object_position_root"], device=runtime.device).repeat(args.num_envs, 1)
    local_quaternion = torch.tensor(plan["object_quaternion_root"], device=runtime.device).repeat(args.num_envs, 1)
    position, quaternion = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, local_position, local_quaternion)
    obj.write_root_pose_to_sim(torch.cat([position, quaternion], dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=runtime.device))
    robot.set_joint_position_target(positions, joint_ids=ids)
    initial_physics = {}
    for name, asset in (("robot", robot), ("object", obj)):
        initial_physics[name] = {field: getattr(asset.root_physx_view, f"get_{field}")().cpu().numpy().tolist()
                                 for field in ("masses", "inertias", "coms", "material_properties")}
    arrays = {name: [] for name in ("joint_position", "joint_velocity", "object_position_root", "jaw_forces")}
    physics_velocity = []
    for target in targets:
        robot.set_joint_position_target(torch.tensor(target, dtype=torch.float32, device=runtime.device).repeat(args.num_envs, 1), joint_ids=ids)
        for _ in range(substeps):
            runtime.scene.write_data_to_sim()
            runtime.sim.step(render=False)
            runtime.scene.update(runtime.physics_dt)
            if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(obj.data.root_state_w).all():
                raise RuntimeError("原生夹爪检查产生无效状态")
            physics_velocity.append(robot.data.joint_vel[:, ids].cpu().numpy().copy())
        object_position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w)
        arrays["joint_position"].append(robot.data.joint_pos[:, ids].cpu().numpy().copy())
        arrays["joint_velocity"].append(robot.data.joint_vel[:, ids].cpu().numpy().copy())
        arrays["object_position_root"].append(object_position.cpu().numpy().copy())
        arrays["jaw_forces"].append(torch.stack([_jaw_force_magnitude(runtime.scene[name])
                                               for name in ("gripper_jaw_contact", "moving_jaw_contact")], dim=-1).cpu().numpy().copy())
    arrays = {name: np.asarray(values) for name, values in arrays.items()}
    if any(not np.isfinite(values).all() for values in arrays.values()) or not np.isfinite(physics_velocity).all():
        raise RuntimeError("原生夹爪记录包含无效数值")
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.output / "trajectory.hdf5", "w") as trace:
        for name, values in arrays.items():
            trace.create_dataset(name, data=values)
        trace.create_dataset("joint_targets", data=targets)
        trace.create_dataset("physics_steps/joint_velocity", data=np.asarray(physics_velocity))
    records = []
    for environment in range(args.num_envs):
        forces = arrays["jaw_forces"][:, environment]
        heights = arrays["object_position_root"][:, environment, 2] - plan["object_position_root"][2]
        bilateral = (forces > .5).all(axis=-1)
        records.append({"environment": environment, "maximum_jaw_forces_n": forces.max(axis=0).tolist(),
                        "bilateral_contact_steps": int(bilateral.sum()), "maximum_object_lift_m": float(heights.max()),
                        "final_object_lift_m": float(heights[-1]), "held_lift_steps": int((bilateral & (heights > .04)).sum()),
                        "maximum_joint_speed_rad_s": np.abs(np.asarray(physics_velocity)[:, environment]).max(axis=0).tolist()})
    report = {"status": "native_scripted_gripper_physics_completed", "environments": records,
              "control_steps": len(targets), "physics_steps_per_environment": len(physics_velocity),
              "physics_dt": runtime.physics_dt, "control_dt": plan["control_dt"], "initial_physics": initial_physics,
              "solver_velocity_iterations": args.velocity_iterations,
              "maximum_depenetration_velocity_m_s": args.max_depenetration_velocity,
              "plan_sha256": digest(args.plan), "trajectory_sha256": digest(args.output / "trajectory.hdf5"),
              "source_code_sha256": digest(Path(__file__)), "controller": "scripted_IK_joint_targets",
              "rl_policy_success_verified": False, "physics_equivalence_verified": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "environments": records}), flush=True)
finally:
    if env is not None:
        env.close()
app.close()
