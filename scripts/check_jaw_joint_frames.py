import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model
from openso101.sim2sim.recorded_physics import COMPONENT_FIELDS, RecordedPhysics


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root = Path(__file__).resolve().parents[1]
folder = root / "outputs/rl_progress/lift_scene_geometry_oriented_verified"
metadata = json.loads((folder / "policy.json").read_text())
model_path = root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml"
usd = root / "outputs/SO-ARM101-USD.usd"
stage = Usd.Stage.Open(str(usd))
joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/so101_new_calib/joints/Jaw"))
if (not joint or str(joint.GetBody0Rel().GetTargets()[0]) != "/so101_new_calib/gripper"
        or str(joint.GetBody1Rel().GetTargets()[0]) != "/so101_new_calib/jaw"):
    raise ValueError("需要经过核查的 USD Jaw 关节连接")
axis_name = joint.GetAxisAttr().Get()
axis = np.eye(3)[("X", "Y", "Z").index(axis_name)]
rotations = []
for attribute in (joint.GetLocalRot0Attr(), joint.GetLocalRot1Attr()):
    quaternion = attribute.Get()
    rotations.append(Rotation.from_quat([quaternion.GetReal(), *quaternion.GetImaginary()], scalar_first=True).as_matrix())
p0, p1 = np.asarray(joint.GetLocalPos0Attr().Get()), np.asarray(joint.GetLocalPos1Attr().Get())
model = build_model(model_path, metadata)
with h5py.File(folder / "isaac_validation.hdf5", "r") as trace:
    fields = {name: trace[name][:1] for name in COMPONENT_FIELDS["bodies"]}
    fields.update({name: trace[name][:1] for name in ("joint_position", "object_position_root", "object_quaternion_root")})
converter = RecordedPhysics(model, metadata, fields, ["bodies"], 0)
parent = converter.body_names.index("gripper")
child = converter.body_names.index("jaw")
data = mujoco.MjData(model)
qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
data.qpos[qids] = fields["joint_position"][0, 0] + JOINT_OFFSETS
records = []
for angle in (0., .1, .125, .35, .8):
    relative_rotation = rotations[0] @ Rotation.from_rotvec(axis * angle).as_matrix() @ rotations[1].T
    relative_position = p0 - relative_rotation @ p1
    ap, ac = converter.frame_rotation[[parent, child]]
    tp, tc = converter.frame_translation[[parent, child]]
    expected_rotation = ap @ relative_rotation @ ac.T
    expected_position = tp + ap @ (relative_position - relative_rotation @ ac.T @ tc)
    data.qpos[qids[-1]] = angle
    mujoco.mj_forward(model, data)
    parent_id, child_id = converter.body_ids[parent], converter.body_ids[child]
    parent_rotation = data.xmat[parent_id].reshape(3, 3)
    actual_rotation = parent_rotation.T @ data.xmat[child_id].reshape(3, 3)
    actual_position = parent_rotation.T @ (data.xpos[child_id] - data.xpos[parent_id])
    position_error = float(np.linalg.norm(actual_position - expected_position))
    rotation_error = float(Rotation.from_matrix(expected_rotation.T @ actual_rotation).magnitude())
    if position_error > 1e-5 or rotation_error > 1e-4:
        raise RuntimeError(f"Jaw 关节坐标检查失败: angle={angle}, position={position_error}, rotation={rotation_error}")
    records.append({"jaw_position_rad": angle, "body_position_error_m": position_error, "body_rotation_error_rad": rotation_error})
result = {"status": "USD_MJCF_jaw_joint_frames_verified", "states": records,
          "usd_sha256": digest(usd), "mjcf_sha256": digest(model_path), "source_code_sha256": digest(Path(__file__)),
          "scope": "authored_USD_joint_transforms_and_real_MuJoCo_forward_kinematics",
          "contact_equivalence_verified": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
