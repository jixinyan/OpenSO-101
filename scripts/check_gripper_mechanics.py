import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares

from openso101.rl.config import digest
from openso101.sim2sim.constrained_drive import ConstrainedImplicitDrive
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model, jaw_forces
from openso101.sim2sim.recorded_physics import COMPONENT_FIELDS, RecordedPhysics


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root = Path(__file__).resolve().parents[1]
folder = root / "outputs/rl_progress/lift_scene_geometry_oriented_verified"
metadata = json.loads((folder / "policy.json").read_text())
validation = json.loads((folder / "validation.json").read_text())
source = folder / "isaac_validation.hdf5"
if digest(source) != validation["trace_sha256"]:
    raise ValueError("夹爪检查的源轨迹校验失败")
model = build_model(root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml", metadata, args.collision_bundle)
data = mujoco.MjData(model)
qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
dids = [int(model.joint(name).dofadr[0]) for name in JOINT_NAMES]
aids = [model.actuator(name).id for name in JOINT_NAMES]
gid = model.body("gripper").id
oid = model.body("object").id
oq = int(model.joint("object_free").qposadr[0])
with h5py.File(source, "r") as trace:
    names = {name for component in COMPONENT_FIELDS.values() for name in component}
    names.update(("joint_position", "object_position_root", "object_quaternion_root", "joint_stiffness", "joint_damping", "joint_vel_limits"))
    fields = {name: trace[name][:1] for name in names}
physics = RecordedPhysics(model, metadata, fields, list(COMPONENT_FIELDS), 0)
physics.apply(data, 0)
kinematics = mujoco.MjData(model)
center_local = np.array([.01, 0, -.09])
object_start = np.array([.02, -.24, metadata["table_height_root"] + .015])


def inverse_kinematics(position, initial):
    def residual(arm):
        kinematics.qpos[qids[:5]] = arm
        kinematics.qpos[qids[5]] = .8
        mujoco.mj_forward(model, kinematics)
        rotation = kinematics.xmat[gid].reshape(3, 3)
        center = kinematics.xpos[gid] + rotation @ center_local
        return np.concatenate([10 * (center - position), rotation[:, 2] - [0, 0, 1]])

    bounds = model.jnt_range[[model.joint(name).id for name in JOINT_NAMES[:5]]]
    solution = least_squares(residual, initial, bounds=(bounds[:, 0] + 1e-4, bounds[:, 1] - 1e-4),
                             ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)
    if not solution.success or np.linalg.norm(residual(solution.x)) > 1e-5:
        raise RuntimeError(f"夹爪姿态 IK 未通过检查: {solution.message}, residual={residual(solution.x)}")
    return solution.x


initial = np.asarray(metadata["default_joint_positions"])[:5] + JOINT_OFFSETS[:5]
approach = inverse_kinematics(object_start + [0, 0, .07], initial)
grasp = inverse_kinematics(object_start, approach)
lift = inverse_kinematics(object_start + [0, 0, .06], grasp)
targets = []
phases = []
previous = np.concatenate([approach, [.8]])
for phase, destination, seconds in (
    ("settle", previous, .5), ("approach", np.concatenate([grasp, [.8]]), 2.),
    ("close", np.concatenate([grasp, [0.]]), 1.5),
    ("lift", np.concatenate([lift, [0.]]), 2.), ("hold", np.concatenate([lift, [0.]]), 1.),
):
    count = round(seconds / metadata["control_dt"])
    for index in range(count):
        targets.append(previous + (destination - previous) * (index + 1) / count)
        phases.append(phase)
    previous = destination
data.qpos[qids] = targets[0]
data.qpos[oq:oq+3] = object_start
data.qpos[oq+3:oq+7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
drive = ConstrainedImplicitDrive(model, aids, qids, dids)
drive.configure(fields["joint_stiffness"][0, 0], fields["joint_damping"][0, 0], fields["joint_vel_limits"][0, 0])
arrays = {name: [] for name in ("joint_position", "joint_velocity", "object_position_root", "jaw_forces", "phase_index")}
physics_arrays = {name: [] for name in ("joint_velocity", "actuator_force", "jaw_forces")}
for step, target in enumerate(targets):
    for _ in range(round(metadata["control_dt"] / model.opt.timestep)):
        drive.apply(data, target)
        mujoco.mj_step(model, data)
        drive.verify(data)
        physics_arrays["joint_velocity"].append(data.qvel[dids].copy())
        physics_arrays["actuator_force"].append(data.actuator_force[aids].copy())
        physics_arrays["jaw_forces"].append(jaw_forces(model, data))
    mujoco.mj_forward(model, data)
    arrays["joint_position"].append(data.qpos[qids].copy() - JOINT_OFFSETS)
    arrays["joint_velocity"].append(data.qvel[dids].copy())
    arrays["object_position_root"].append(data.xpos[oid].copy())
    arrays["jaw_forces"].append(jaw_forces(model, data))
    arrays["phase_index"].append(("settle", "approach", "close", "lift", "hold").index(phases[step]))
arrays = {name: np.asarray(values) for name, values in arrays.items()}
forces = arrays["jaw_forces"]
heights = arrays["object_position_root"][:, 2] - object_start[2]
bilateral = (forces > metadata["grasp_force_threshold_newtons"]).all(axis=-1)
args.output.mkdir(parents=True, exist_ok=False)
with h5py.File(args.output / "trajectory.hdf5", "w") as trace:
    for name, values in arrays.items():
        trace.create_dataset(name, data=values)
    trace.create_dataset("joint_targets", data=np.asarray(targets) - JOINT_OFFSETS)
    for name, values in physics_arrays.items():
        trace.create_dataset(f"physics_steps/{name}", data=np.asarray(values))
plan = {"initial_joint_position": (targets[0] - JOINT_OFFSETS).tolist(), "object_position_root": object_start.tolist(),
        "object_quaternion_root": [1., 0., 0., 0.], "control_dt": metadata["control_dt"],
        "joint_targets": (np.asarray(targets) - JOINT_OFFSETS).tolist(), "phases": phases,
        "grasp_center_local": center_local.tolist()}
(args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
report = {"status": "scripted_gripper_physics_completed", "control_steps": len(targets),
          "maximum_jaw_forces_n": forces.max(axis=0).tolist(), "bilateral_contact_steps": int(bilateral.sum()),
          "maximum_object_lift_m": float(heights.max()), "final_object_lift_m": float(heights[-1]),
          "held_lift_steps": int((bilateral & (heights > .04)).sum()), "drive": drive.report(),
          "physics": physics.report(), "source_trace_sha256": digest(source),
          "trajectory_sha256": digest(args.output / "trajectory.hdf5"), "plan_sha256": digest(args.output / "plan.json"),
          "source_code_sha256": digest(Path(__file__)), "controller": "scripted_IK_joint_targets",
          "rl_policy_success_verified": False, "physics_equivalence_verified": False}
report["collision_bundle_sha256"] = digest(args.collision_bundle / "manifest.json") if args.collision_bundle else None
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
