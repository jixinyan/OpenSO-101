import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from openso101.scenes.models import file_digest

parser = argparse.ArgumentParser()
parser.add_argument("policy", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--high-limit", type=float, default=1000)
args = parser.parse_args()
if args.steps <= 1 or not np.isfinite(args.high_limit) or args.high_limit <= 2:
    raise ValueError("steps 必须大于 1，high-limit 必须大于 2 且有限")
if args.output.exists():
    raise FileExistsError(args.output)
metadata = json.loads((args.policy / "policy.json").read_text())
trace_path = args.policy / "isaac_validation.hdf5"
validation = json.loads((args.policy / "validation.json").read_text())
if validation["trace_sha256"] != file_digest(trace_path):
    raise ValueError("源轨迹 SHA256 与实际验证报告不一致")
with h5py.File(trace_path, "r") as trace:
    fields = {name: trace[name][:args.steps] for name in (
        "joint_position", "joint_velocity", "joint_targets", "joint_stiffness", "joint_damping", "joint_vel_limits",
        "object_position_root", "object_quaternion_root", "object_linear_velocity_root", "object_angular_velocity_root",
        "terminated", "truncated",
    )}
if len(fields["joint_position"]) != args.steps or any(not np.isfinite(value).all() for value in fields.values()):
    raise ValueError("源轨迹数量不足或包含无效数据")
if (fields["terminated"][:-1] | fields["truncated"][:-1]).any():
    raise ValueError("请求范围包含源 episode 的重置边界")
pairs = fields["joint_position"].shape[1]
if fields["joint_position"].shape != (args.steps, pairs, 6):
    raise ValueError("源关节数据需要 steps×environments×6 形状")
args.task = metadata["task_id"]
args.task_profile = metadata["task_profile"]
args.num_envs = pairs * 2
args.seed = 42
args.with_cameras = False

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    import torch
    from isaaclab.utils.math import combine_frame_transforms, quat_apply

    from openso101.rl.execution import build_environment
    from openso101.robots.so101.constants import SO101_SIM_JOINT_NAMES

    env = build_environment(args, training=False)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    obj = unwrapped.scene["object"]
    joint_ids = [robot.joint_names.index(name) for name in SO101_SIM_JOINT_NAMES]
    if not np.isclose(unwrapped.step_dt, metadata["control_dt"]):
        raise ValueError("原生控制周期与源记录不一致")
    physics = {}
    indices = torch.arange(args.num_envs, dtype=torch.int32, device="cpu")
    for asset_name, asset in (("robot", robot), ("object", obj)):
        view = asset.root_physx_view
        physics[asset_name] = {}
        for name in ("masses", "inertias", "material_properties"):
            values = getattr(view, f"get_{name}")().clone()
            values[pairs:] = values[:pairs].clone()
            getattr(view, f"set_{name}")(values, indices)
            actual = getattr(view, f"get_{name}")()
            if not torch.equal(actual[:pairs], actual[pairs:]):
                raise ValueError(f"配对实际物理参数不一致: {asset_name}/{name}")
            physics[asset_name][name] = actual[:pairs].cpu().numpy()
        coms = view.get_coms()
        if not torch.equal(coms[:pairs], coms[pairs:]):
            raise ValueError(f"配对 COM 不一致: {asset_name}")
        physics[asset_name]["coms"] = coms[:pairs].cpu().numpy()
    for name in ("joint_armature", "joint_friction_coeff", "joint_pos_limits", "joint_effort_limits"):
        values = getattr(robot.data, name)
        if not torch.equal(values[:pairs], values[pairs:]):
            raise ValueError(f"配对关节参数不一致: {name}")
        physics[name] = values[:pairs].cpu().numpy()

    def paired(value):
        tensor = torch.as_tensor(value, device=unwrapped.device, dtype=torch.float32)
        return torch.cat((tensor, tensor), dim=0)

    robot.write_joint_state_to_sim(paired(fields["joint_position"][0]), paired(fields["joint_velocity"][0]), joint_ids=joint_ids)
    root = obj.data.default_root_state.clone()
    root[:, :3], root[:, 3:7] = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        paired(fields["object_position_root"][0]), paired(fields["object_quaternion_root"][0]),
    )
    root[:, 7:10] = quat_apply(robot.data.root_quat_w, paired(fields["object_linear_velocity_root"][0]))
    root[:, 10:13] = quat_apply(robot.data.root_quat_w, paired(fields["object_angular_velocity_root"][0]))
    obj.write_root_state_to_sim(root)
    limits = paired(fields["joint_vel_limits"][0])
    limits[pairs:] = args.high_limit
    robot.write_joint_velocity_limit_to_sim(limits, joint_ids=joint_ids)
    if not torch.equal(robot.data.joint_vel_limits[:, joint_ids], limits) or not torch.equal(robot.root_physx_view.get_dof_max_velocities()[:, joint_ids], limits.cpu()):
        raise ValueError("实际速度限制写入检查失败")
    initial_position_error = float(torch.max(torch.abs(robot.data.joint_pos[:, joint_ids] - paired(fields["joint_position"][0]))))
    initial_velocity_error = float(torch.max(torch.abs(robot.data.joint_vel[:, joint_ids] - paired(fields["joint_velocity"][0]))))
    if max(initial_position_error, initial_velocity_error) > 1e-6:
        raise ValueError("初始关节状态与源记录不一致")
    buffers = {name: [] for name in ("joint_position", "joint_velocity", "object_position_world")}
    for step in range(args.steps):
        buffers["joint_position"].append(robot.data.joint_pos[:, joint_ids].cpu().numpy().copy())
        buffers["joint_velocity"].append(robot.data.joint_vel[:, joint_ids].cpu().numpy().copy())
        buffers["object_position_world"].append(obj.data.root_pos_w.cpu().numpy().copy())
        stiffness = paired(fields["joint_stiffness"][step])
        damping = paired(fields["joint_damping"][step])
        robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids)
        robot.write_joint_damping_to_sim(damping, joint_ids=joint_ids)
        if not torch.equal(robot.root_physx_view.get_dof_stiffnesses()[:, joint_ids], stiffness.cpu()) or not torch.equal(robot.root_physx_view.get_dof_dampings()[:, joint_ids], damping.cpu()):
            raise ValueError("实际 PD 参数写入检查失败")
        robot.set_joint_position_target(paired(fields["joint_targets"][step]), joint_ids=joint_ids)
        for _ in range(unwrapped.cfg.decimation):
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(unwrapped.physics_dt)
            if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(robot.data.joint_vel).all() or not torch.isfinite(obj.data.root_state_w).all():
                raise RuntimeError("原生速度实验产生无效状态")
    arrays = {name: np.asarray(value) for name, value in buffers.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.output / "velocity_limit.hdf5", "w") as result:
        for name, value in arrays.items():
            result.create_dataset(name, data=value)
        for asset_name in ("robot", "object"):
            for name, value in physics[asset_name].items():
                result.create_dataset(f"physics/{asset_name}/{name}", data=value)
        for name in ("joint_armature", "joint_friction_coeff", "joint_pos_limits", "joint_effort_limits"):
            result.create_dataset(f"physics/{name}", data=physics[name])
        for name in ("joint_targets", "joint_stiffness", "joint_damping"):
            result.create_dataset(f"input/{name}", data=fields[name])
        result.create_dataset("velocity_limits", data=limits.cpu().numpy())
    difference = arrays["joint_position"][:, pairs:] - arrays["joint_position"][:, :pairs]
    records = []
    for index in range(pairs):
        records.append({
            "source_env_index": index,
            "limited_max_joint_speed_rad_s": np.max(np.abs(arrays["joint_velocity"][:, index]), axis=0).tolist(),
            "high_limit_max_joint_speed_rad_s": np.max(np.abs(arrays["joint_velocity"][:, index + pairs]), axis=0).tolist(),
            "paired_max_joint_position_difference_rad": np.max(np.abs(difference[:, index]), axis=0).tolist(),
            "limited_source_max_position_difference_rad": np.max(np.abs(arrays["joint_position"][:, index] - fields["joint_position"][:, index]), axis=0).tolist(),
        })
    report = {
        "status": "native_isaac_velocity_limit_experiment_completed", "task": args.task,
        "pairs": pairs, "steps": args.steps, "control_dt": unwrapped.step_dt,
        "physics_dt": unwrapped.physics_dt, "high_limit_rad_s": args.high_limit,
        "initial_position_error_rad": initial_position_error, "initial_velocity_error_rad_s": initial_velocity_error,
        "paired_physics_verified": True, "paired_pd_verified": True,
        "environments": records, "source_trace_sha256": file_digest(trace_path),
        "source_code_sha256": file_digest(Path(__file__)),
        "trajectory_sha256": file_digest(args.output / "velocity_limit.hdf5"),
        "source_physics_reproduction_verified": False, "physics_equivalence_verified": False,
        "task_success_verified": False,
        "scope": "原生 PhysX 求解器速度限制的配对比较，使用相同场景与记录的实际 PD 和动作",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
finally:
    if env is not None:
        env.close()
    app.close()
