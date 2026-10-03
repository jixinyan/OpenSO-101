import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .mujoco import JOINT_NAMES, JOINT_OFFSETS


def plan_collision_grasp(model, states, environment, rng):
    data = mujoco.MjData(model)
    ids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    gripper = model.body("gripper").id
    object_qpos = int(model.joint("object_free").qposadr[0])
    start = np.asarray(environment["object_position_root"])
    data.qpos[object_qpos:object_qpos + 3] = start
    data.qpos[object_qpos + 3:object_qpos + 7] = environment["object_quaternion_root"]
    initial = np.asarray(environment["joint_position"][:5]) + JOINT_OFFSETS[:5]
    limits = np.asarray(states["soft_joint_limits"])[:5] + JOINT_OFFSETS[:5, None]
    lower, upper = limits[:, 0] + 1e-5, limits[:, 1] - 1e-5
    center = np.array([.01, 0., -.09])
    goal = np.asarray(environment["goal_position_root"])
    object_offset = .25 * states["planner_physics"]["object_size"][2]
    lift_radius = states["planner_physics"]["task_goal_radius"] - object_offset - .005
    if not np.isfinite(lift_radius) or lift_radius <= .003:
        raise ValueError("原生任务目标范围无法容纳抓取中心与跟踪误差")
    positions = [start + [0., 0., .03], start + [0., 0., .25 * states["planner_physics"]["object_size"][2]], goal]

    def pose(arm, jaw=.8, allow_grasp_contact=False):
        data.qpos[ids[:5]] = arm
        data.qpos[ids[5]] = jaw
        mujoco.mj_forward(model, data)
        rotation = data.xmat[gripper].reshape(3, 3).copy()
        position = data.xpos[gripper] + rotation @ center
        depth = 0.
        for contact in data.contact:
            geometries = [model.geom(index).name for index in (contact.geom1, contact.geom2)]
            bodies = [model.body(model.geom_bodyid[index]).name for index in (contact.geom1, contact.geom2)]
            if "table" in geometries and set(bodies) <= {"world", "base", "object"}:
                continue
            if allow_grasp_contact and "object" in bodies and any(
                    body in ("gripper", "moving_jaw_so101_v1") for body in bodies):
                continue
            depth = max(depth, -float(contact.dist))
        return position, rotation, depth

    def coupled_residual(joints):
        poses = [pose(arm, jaw=0. if index == 2 else .8, allow_grasp_contact=index == 2)
                 for index, arm in enumerate(joints.reshape(3, 5))]
        lift_error = poses[2][0] - goal
        lift_violation = lift_error * max(0., 1 - lift_radius / max(np.linalg.norm(lift_error), 1e-12))
        return np.concatenate((20 * (poses[0][0] - positions[0]), 20 * (poses[1][0] - positions[1]),
                               20 * lift_violation, .01 * lift_error,
                               (poses[0][1] - poses[1][1]).ravel(),
                               (poses[1][1] - poses[2][1]).ravel(),
                               [.05 * (1 - value[1][2, 2]) for value in poses],
                               [100 * value[2] for value in poses]))

    starts = [np.tile(np.clip(initial, lower, upper), 3)]
    starts.extend(np.tile(arm, 3) for arm in rng.uniform(lower, upper, size=(31, 5)))
    solutions = [least_squares(coupled_residual, guess, bounds=(np.tile(lower, 3), np.tile(upper, 3)),
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000) for guess in starts]

    def feasible(solution):
        poses = [pose(arm, jaw=0. if index == 2 else .8, allow_grasp_contact=index == 2)
                 for index, arm in enumerate(solution.x.reshape(3, 5))]
        return (solution.success and all(np.linalg.norm(value[0] - target) <= (.003 if index < 2 else lift_radius)
                                        and value[2] <= .0001
                                        for index, (value, target) in enumerate(zip(poses, positions, strict=True)))
                and np.linalg.norm(poses[0][1] - poses[1][1]) <= .02
                and np.linalg.norm(poses[1][1] - poses[2][1]) <= .02)

    def path_safe(start_arm, end_arm, jaw=.8, allow_grasp_contact=False):
        return all(pose(point, jaw, allow_grasp_contact)[2] <= .0001
                   for point in np.linspace(start_arm, end_arm, 65)[1:])

    def cartesian_path(start_arm, end_arm, jaw=.8, allow_grasp_contact=False):
        first, last = pose(start_arm, jaw, allow_grasp_contact), pose(end_arm, jaw, allow_grasp_contact)
        fractions = np.linspace(0., 1., 65)[1:]
        rotations = Slerp([0., 1.], Rotation.from_matrix([first[1], last[1]]))(fractions).as_matrix()
        previous = start_arm
        path = []
        accepted = True
        for fraction, target_rotation in zip(fractions, rotations, strict=True):
            target_position = first[0] + fraction * (last[0] - first[0])

            def residual(arm):
                position, rotation, depth = pose(arm, jaw, allow_grasp_contact)
                return np.concatenate((20 * (position - target_position), (rotation - target_rotation).ravel(),
                                       [100 * depth], .0001 * (arm - previous)))

            solution = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper),
                                     ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=100)
            position, rotation, depth = pose(solution.x, jaw, allow_grasp_contact)
            accepted &= bool(solution.success and np.linalg.norm(position - target_position) <= .003
                             and np.linalg.norm(rotation - target_rotation) <= .02 and depth <= .0001)
            path.append(solution.x)
            previous = solution.x
        return np.asarray(path), accepted

    candidates = sorted((item for item in solutions if feasible(item) and path_safe(initial, item.x[:5])),
                        key=lambda item: np.linalg.norm(item.x[:5] - initial)
                        + np.linalg.norm(np.diff(item.x.reshape(3, 5), axis=0)))
    result = min(solutions, key=lambda item: np.linalg.norm(coupled_residual(item.x)))
    grasp_path = None
    lift_path = None
    paths_accepted = False
    for candidate in candidates:
        candidate_path, valid = cartesian_path(candidate.x[:5], candidate.x[5:10])
        candidate_lift, lift_valid = cartesian_path(candidate_path[-1], candidate.x[10:], jaw=0., allow_grasp_contact=True)
        if valid and lift_valid:
            result, grasp_path, lift_path, paths_accepted = candidate, candidate_path, candidate_lift, True
            break
    arms = [result.x[:5], grasp_path[-1] if grasp_path is not None else result.x[5:10],
            lift_path[-1] if lift_path is not None else result.x[10:]]
    targets = []
    previous = initial
    for index, (name, arm, target) in enumerate(zip(("approach", "grasp", "lift"), arms, positions, strict=True)):
        path = (lift_path if name == "lift" and lift_path is not None else grasp_path if name == "grasp" and grasp_path is not None
                else np.linspace(previous, arm, 65)[1:])
        samples = [pose(point, jaw=0. if name == "lift" else .8, allow_grasp_contact=name == "lift") for point in path]
        position, rotation, depth = samples[-1]
        maximum_depth = max(value[2] for value in samples)
        accepted = paths_accepted and maximum_depth <= .0001
        targets.append({"phase": name, "target_position_root": target.tolist(),
                        "joint_position": (arm - JOINT_OFFSETS[:5]).tolist(),
                        "path_joint_positions": (path - JOINT_OFFSETS[:5]).tolist(),
                        "position_error_m": float(np.linalg.norm(position - target)),
                        "gripper_inclination_rad": float(np.arccos(np.clip(rotation[2, 2], -1, 1))),
                        "maximum_penetration_m": depth, "sampled_path_maximum_penetration_m": maximum_depth,
                        "path_samples": len(samples), "accepted": bool(accepted), "attempts": len(solutions)})
        previous = arm
    return {"environment": environment["environment"], "targets": targets,
            "accepted": all(item["accepted"] for item in targets),
            "approach_grasp_rotation_difference": float(np.linalg.norm(pose(arms[0])[1] - pose(arms[1])[1])),
            "grasp_lift_rotation_difference": float(np.linalg.norm(pose(arms[1])[1] - pose(arms[2])[1])),
            "coupled_waypoints": 3, "native_task_goal_radius_m": states["planner_physics"]["task_goal_radius"],
            "planned_lift_center_radius_m": lift_radius}
