import json
from pathlib import Path

import h5py
import matplotlib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openso101.rl.config import digest
from openso101.rl.portable import PortablePolicy
from .mujoco import JOINT_NAMES, JOINT_OFFSETS, check_kinematics


def _read_fields(group, steps):
    fields = {}
    for name, width in (("joint_position", 6), ("joint_targets", 6), ("raw_action", 6),
                        ("object_position_root", 3), ("jaw_forces", 2)):
        if group[name].shape != (steps, width):
            raise ValueError(f"实际轨迹的字段形状无效：{name}")
        values = group[name][:]
        if not np.isfinite(values).all():
            raise ValueError(f"实际轨迹的字段形状或数值无效：{name}")
        fields[name] = values
    if (fields["jaw_forces"] < 0).any():
        raise ValueError("jaw_forces 需要非负接触力大小")
    return fields


def _geometry(model, positions, objects):
    data = mujoco.MjData(model)
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    gripper = model.body("gripper").id
    local_positions = []
    for joints, position in zip(positions, objects, strict=True):
        data.qpos[addresses] = joints + JOINT_OFFSETS
        mujoco.mj_forward(model, data)
        rotation = Rotation.from_quat(data.xquat[gripper], scalar_first=True)
        local_positions.append(rotation.inv().apply(position - data.xpos[gripper]))
    local = np.asarray(local_positions)
    displacement = local - [.01, 0., -.09]
    return {
        "ee_object_distance_m": np.linalg.norm(displacement, axis=-1),
        "grasp_alignment": np.exp(-np.sum((displacement / [.04, .025, .03]) ** 2, axis=-1)),
    }


def _metrics(policy, fields, geometry, threshold, action_tolerance):
    decoded = policy.joint_targets(fields["raw_action"], enforce_limits=False,
                                   joint_position=fields["joint_position"]).numpy()
    error = float(np.max(np.abs(decoded - fields["joint_targets"])))
    if error > action_tolerance:
        raise ValueError("实际 action 与保存的 joint_targets 不一致")
    if (fields["jaw_forces"] < 0).any():
        raise ValueError("jaw_forces 需要非负接触力大小")
    bilateral = np.minimum(fields["jaw_forces"][:, 0], fields["jaw_forces"][:, 1])
    reached = geometry["ee_object_distance_m"] < .08
    contacts = bilateral > threshold
    return {
        "frames": len(bilateral), "action_conversion_maximum_error_rad": error,
        "action_conversion_tolerance_rad": action_tolerance,
        "jaw_target_range_rad": [float(fields["joint_targets"][:, -1].min()),
                                 float(fields["joint_targets"][:, -1].max())],
        "jaw_actual_range_rad": [float(fields["joint_position"][:, -1].min()),
                                 float(fields["joint_position"][:, -1].max())],
        "joint_target_tracking_rmse_rad": np.sqrt(np.mean(
            (fields["joint_targets"] - fields["joint_position"]) ** 2, axis=0)).tolist(),
        "reached_frames": int(reached.sum()), "bilateral_contact_frames": int(contacts.sum()),
        "reached_bilateral_contact_frames": int((reached & contacts).sum()),
        "maximum_bilateral_force_n": float(bilateral.max()),
        "maximum_grasp_alignment": float(geometry["grasp_alignment"].max()),
        "minimum_ee_object_distance_m": float(geometry["ee_object_distance_m"].min()),
        "object_height_range_root_m": [float(fields["object_position_root"][:, 2].min()),
                                       float(fields["object_position_root"][:, 2].max())],
    }


def _plot(cases, control_dt, threshold, output):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(len(cases), 3, figsize=(15, 3 * len(cases)), squeeze=False)
    for index, (record, sources) in enumerate(cases):
        for label, color in (("isaac", "tab:blue"), ("mujoco", "tab:orange")):
            fields, geometry = sources[label]
            times = np.arange(len(fields["jaw_forces"])) * control_dt
            axes[index, 0].plot(times, fields["joint_position"][:, -1], color=color, label=f"{label} actual")
            axes[index, 0].plot(times, fields["joint_targets"][:, -1], color=color,
                                linestyle="--", alpha=.65, label=f"{label} target")
            axes[index, 1].plot(times, fields["jaw_forces"].min(axis=-1), color=color, label=label)
            axes[index, 2].plot(times, geometry["ee_object_distance_m"], color=color, label=label)
        axes[index, 0].set_ylabel(f"Env {record['source_env_index']} jaw (rad)")
        axes[index, 1].set_ylabel("Bilateral contact (N)")
        axes[index, 1].axhline(threshold, color="black", linestyle=":", label="grasp threshold")
        axes[index, 2].set_ylabel("End effector to object (m)")
        axes[index, 2].axhline(.08, color="black", linestyle=":", label="reach threshold")
        for axis in axes[index]:
            axis.set_xlabel("Control time (s)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.suptitle("Saved policy: actual Isaac and MuJoCo trajectories\n"
                    "Same initial environments; independent control trajectories")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def diagnose(args):
    policy_folder = Path(args.policy).resolve()
    evaluation = Path(args.evaluation).resolve()
    robot_model = Path(args.robot_model).resolve()
    output = Path(args.output).resolve()
    if Path(args.output).is_symlink() or output.exists():
        raise FileExistsError(output)
    for source in (policy_folder, evaluation, robot_model.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("diagnostics 输出目录需要与来源目录独立")
    policy = PortablePolicy(policy_folder)
    metadata = policy.metadata
    report_path = evaluation / "report.json"
    source_report = json.loads(report_path.read_text())
    report_sha256 = digest(report_path)
    trajectory_path = evaluation / "trajectory.hdf5"
    native_path = policy_folder / "isaac_validation.hdf5"
    sources = {
        "policy_metadata_sha256": policy_folder / "policy.json",
        "policy_sha256": policy_folder / "policy.pt",
        "robot_model_sha256": robot_model,
        "isaac_trace_sha256": native_path,
        "trajectory_sha256": trajectory_path,
    }
    for name, path in sources.items():
        if digest(path) != source_report[name]:
            raise ValueError(f"diagnostics 来源 SHA256 不一致：{name}")
    if (source_report["status"] != "mujoco_policy_evaluation_completed"
            or source_report["task"] != metadata["task_id"]
            or source_report["control_dt"] != metadata["control_dt"]):
        raise ValueError("需要相同策略、任务与控制周期的实际 MuJoCo 评估")
    validation = json.loads((policy_folder / "validation.json").read_text())
    if digest(native_path) != validation["trace_sha256"]:
        raise ValueError("Isaac 来源轨迹 SHA256 不一致")
    if (validation["status"] != "portable_policy_numerically_verified_in_isaac"
            or any(not np.isfinite(value) or not 0 <= value <= 1e-5
                   for value in validation["maximum_errors"].values())):
        raise ValueError("Isaac 来源策略需要通过原生数值验证")
    model = mujoco.MjModel.from_xml_path(str(robot_model))
    threshold = metadata["grasp_force_threshold_newtons"]
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("接触力阈值需要正数有限数值")
    records = source_report["episodes"]
    indices = [record["initial_env_index"] for record in records]
    if not records or len(set(indices)) != len(indices):
        raise ValueError("实际评估需要独立的来源环境编号")
    cases = []
    with h5py.File(native_path, "r") as native, h5py.File(trajectory_path, "r") as trajectory:
        if native.attrs["state_sampling"] != "before_control_step":
            raise ValueError("Isaac 轨迹需要记录每个控制步骤之前的状态")
        kinematics = check_kinematics(model, native)
        environments = native["joint_position"].shape[1]
        if any(type(index) is not int or not 0 <= index < environments for index in indices):
            raise ValueError("来源环境编号超过实际 Isaac 轨迹范围")
        if set(trajectory) != {f"episode_{index:06d}" for index in range(len(records))}:
            raise ValueError("MuJoCo 轨迹与评估的 episode 数量不一致")
        for episode, record in enumerate(records):
            env_index = record["initial_env_index"]
            boundaries = np.flatnonzero(native["terminated"][:, env_index] | native["truncated"][:, env_index])
            steps = int(boundaries[0]) + 1 if len(boundaries) else len(native["joint_position"])
            native_fields = {name: native[name][:steps, env_index] for name in (
                "joint_position", "joint_targets", "raw_action", "object_position_root", "jaw_forces")}
            for name, width in (("joint_position", 6), ("joint_targets", 6), ("raw_action", 6),
                                ("object_position_root", 3), ("jaw_forces", 2)):
                if native_fields[name].shape != (steps, width) or not np.isfinite(native_fields[name]).all():
                    raise ValueError(f"Isaac 字段形状或数值无效：{name}")
            target_group = trajectory[f"episode_{episode:06d}"]
            target_steps = record["steps"]
            if type(target_steps) is not int or target_steps <= 0 or len(target_group["joint_position"]) != target_steps:
                raise ValueError("MuJoCo 轨迹帧数与实际报告不一致")
            target_fields = _read_fields(target_group, target_steps)
            native_geometry = _geometry(model, native_fields["joint_position"], native_fields["object_position_root"])
            distance_error = float(np.max(np.abs(native_geometry["ee_object_distance_m"]
                                                 - native["ee_object_distance"][:steps, env_index])))
            if distance_error > 1e-5:
                raise ValueError("末端参考点未通过实际 Isaac ee_object_distance 检查")
            target_geometry = _geometry(model, target_fields["joint_position"], target_fields["object_position_root"])
            initial_error = float(np.max(np.abs(native_fields["joint_position"][0] - target_fields["joint_position"][0])))
            initial_object_error = float(np.linalg.norm(native_fields["object_position_root"][0]
                                                        - target_fields["object_position_root"][0]))
            if initial_error > 1e-6 or initial_object_error > 1e-6:
                raise ValueError("Isaac 与 MuJoCo 的实际初始状态不一致")
            result = {"source_env_index": env_index, "mujoco_task_success": record["success"],
                      "isaac_first_episode_boundary_recorded": bool(len(boundaries)),
                      "initial_joint_error_rad": initial_error, "initial_object_error_m": initial_object_error,
                      "native_ee_distance_maximum_error_m": distance_error,
                      "isaac": _metrics(policy, native_fields, native_geometry, threshold, 1e-5),
                      "mujoco": _metrics(policy, target_fields, target_geometry, threshold, 1e-6)}
            cases.append((result, {"isaac": (native_fields, native_geometry),
                                   "mujoco": (target_fields, target_geometry)}))
    for name, path in sources.items():
        if digest(path) != source_report[name]:
            raise ValueError(f"diagnostics 检查期间来源内容发生变化：{name}")
    if digest(report_path) != report_sha256:
        raise ValueError("diagnostics 检查期间评估报告发生变化")
    output.mkdir(parents=True, exist_ok=False)
    _plot(cases, metadata["control_dt"], threshold, output / "diagnostics.png")
    result = {"status": "actual_sim2sim_diagnostics_verified", "task": metadata["task_id"],
              "task_profile": metadata["task_profile"], "environments": [record for record, _ in cases],
              "kinematics": kinematics, "control_dt": metadata["control_dt"],
              "grasp_force_threshold_newtons": threshold,
              "inputs": {name: digest(path) for name, path in sources.items()},
              "evaluation_report_sha256": report_sha256,
              "source_sha256": digest(Path(__file__)), "figure_sha256": digest(output / "diagnostics.png"),
              "training_started": False, "gpu_tests_started": False,
              "physics_equivalence_verified": False, "rl_task_success_verified": False}
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0
