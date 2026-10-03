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
    names = ["approach", "grasp", "lift"]
    pick_place = states["task"] == "OpenSO101-PickPlace-v0"
    if pick_place:
        positions.extend((np.asarray(environment["carry_goal_position_root"]),
                          np.asarray(environment["place_goal_position_root"]) + [0., 0., object_offset]))
        names.extend(("carry", "place"))
    waypoint_count = len(positions)

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
        poses = [pose(arm, jaw=0. if index >= 2 else .8, allow_grasp_contact=index >= 2)
                 for index, arm in enumerate(joints.reshape(waypoint_count, 5))]
        residuals = [20 * (poses[0][0] - positions[0]), 20 * (poses[1][0] - positions[1])]
        for index in range(2, waypoint_count):
            error = poses[index][0] - positions[index]
            if names[index] in ("lift", "carry"):
                violation = error * max(0., 1 - (lift_radius - .001) / max(np.linalg.norm(error), 1e-12))
                residuals.extend((20 * violation, .01 * error))
            else:
                residuals.append(20 * error)
        residuals.extend((poses[index][1] - poses[index + 1][1]).ravel() for index in range(waypoint_count - 1))
        residuals.extend(([.05 * (1 - value[1][2, 2]) for value in poses], [100 * value[2] for value in poses]))
        return np.concatenate(residuals)

    starts = [np.tile(np.clip(initial, lower, upper), waypoint_count)]
    starts.extend(np.tile(arm, waypoint_count) for arm in rng.uniform(lower, upper, size=(31, 5)))
    solutions = [least_squares(coupled_residual, guess, bounds=(np.tile(lower, waypoint_count), np.tile(upper, waypoint_count)),
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000) for guess in starts]

    def feasible(solution):
        poses = [pose(arm, jaw=0. if index >= 2 else .8, allow_grasp_contact=index >= 2)
                 for index, arm in enumerate(solution.x.reshape(waypoint_count, 5))]
        return (solution.success and all(np.linalg.norm(value[0] - target) <= (lift_radius if names[index] in ("lift", "carry") else .003)
                                        and value[2] <= .0001
                                        for index, (value, target) in enumerate(zip(poses, positions, strict=True)))
                and all(np.linalg.norm(poses[index][1] - poses[index + 1][1]) <= .02 for index in range(waypoint_count - 1)))

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
                        + np.linalg.norm(np.diff(item.x.reshape(waypoint_count, 5), axis=0)))
    result = min(solutions, key=lambda item: np.linalg.norm(coupled_residual(item.x)))
    planned_paths = None
    paths_accepted = False
    for candidate in candidates:
        candidate_arms = candidate.x.reshape(waypoint_count, 5)
        candidate_paths = [np.linspace(initial, candidate_arms[0], 65)[1:]]
        valid = True
        for index in range(1, waypoint_count):
            path, accepted = cartesian_path(candidate_paths[-1][-1], candidate_arms[index],
                                            jaw=0. if index >= 2 else .8, allow_grasp_contact=index >= 2)
            valid &= accepted
            candidate_paths.append(path)
        if valid:
            result, planned_paths, paths_accepted = candidate, candidate_paths, True
            break
    arms = [path[-1] for path in planned_paths] if planned_paths is not None else result.x.reshape(waypoint_count, 5)
    targets = []
    previous = initial
    for index, (name, arm, target) in enumerate(zip(names, arms, positions, strict=True)):
        path = planned_paths[index] if planned_paths is not None else np.linspace(previous, arm, 65)[1:]
        samples = [pose(point, jaw=0. if index >= 2 else .8, allow_grasp_contact=index >= 2) for point in path]
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
            "coupled_waypoints": waypoint_count, "native_task_goal_radius_m": states["planner_physics"]["task_goal_radius"],
            "planned_lift_center_radius_m": lift_radius}
