import mujoco
import numpy as np
from scipy.optimize import least_squares

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
    positions = [start + [0., 0., .03], start + [0., 0., .25 * states["planner_physics"]["object_size"][2]]]

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
        poses = [pose(arm) for arm in joints.reshape(2, 5)]
        return np.concatenate((*(20 * (value[0] - target) for value, target in zip(poses, positions, strict=True)),
                               (poses[0][1] - poses[1][1]).ravel(),
                               [max(0., np.cos(np.pi / 4) - value[1][2, 2]) for value in poses],
                               [.05 * (1 - value[1][2, 2]) for value in poses],
                               [100 * value[2] for value in poses]))

    starts = [np.tile(np.clip(initial, lower, upper), 2)]
    starts.extend(np.tile(arm, 2) for arm in rng.uniform(lower, upper, size=(31, 5)))
    solutions = [least_squares(coupled_residual, guess, bounds=(np.tile(lower, 2), np.tile(upper, 2)),
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=1000) for guess in starts]

    def feasible(solution):
        poses = [pose(arm) for arm in solution.x.reshape(2, 5)]
        return (solution.success and all(np.linalg.norm(value[0] - target) <= .003 and value[2] <= .0001
                                        and value[1][2, 2] >= np.cos(np.pi / 4) - .01
                                        for value, target in zip(poses, positions, strict=True))
                and np.linalg.norm(poses[0][1] - poses[1][1]) <= .02)

    def path_safe(start_arm, end_arm, jaw=.8, allow_grasp_contact=False):
        return all(pose(point, jaw, allow_grasp_contact)[2] <= .0001
                   for point in np.linspace(start_arm, end_arm, 65)[1:])

    accepted_solutions = [item for item in solutions if feasible(item)
                          and path_safe(initial, item.x[:5]) and path_safe(item.x[:5], item.x[5:])]
    result = min(accepted_solutions, key=lambda item: np.linalg.norm(item.x[:5] - initial)
                 + np.linalg.norm(item.x[5:] - item.x[:5])) if accepted_solutions else min(
                     solutions, key=lambda item: np.linalg.norm(coupled_residual(item.x)))
    arms = list(result.x.reshape(2, 5))
    previous = arms[-1]
    goal = np.asarray(environment["goal_position_root"])

    def lift_residual(arm):
        position, rotation, depth = pose(arm, jaw=0., allow_grasp_contact=True)
        return np.concatenate((20 * (position - goal), .01 * (arm - previous), [100 * depth]))

    lift_starts = [np.clip(previous, lower, upper), *rng.uniform(lower, upper, size=(15, 5))]
    lift_solutions = [least_squares(lift_residual, guess, bounds=(lower, upper),
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500) for guess in lift_starts]
    lift_feasible = [item for item in lift_solutions if item.success and np.linalg.norm(lift_residual(item.x)[:3]) <= .06
                    and lift_residual(item.x)[-1] <= .01
                    and path_safe(previous, item.x, jaw=0., allow_grasp_contact=True)]
    lift = min(lift_feasible, key=lambda item: np.linalg.norm(item.x - previous)) if lift_feasible else min(
        lift_solutions, key=lambda item: np.linalg.norm(lift_residual(item.x)))
    arms.append(lift.x)
    targets = []
    previous = initial
    for index, (name, arm, target) in enumerate(zip(("approach", "grasp", "lift"), arms, [*positions, goal], strict=True)):
        path = np.linspace(previous, arm, 65)[1:]
        samples = [pose(point, jaw=0. if name == "lift" else .8, allow_grasp_contact=name == "lift") for point in path]
        position, rotation, depth = samples[-1]
        maximum_depth = max(value[2] for value in samples)
        accepted = (bool(accepted_solutions) if index < 2 else bool(lift_feasible)) and maximum_depth <= .0001
        targets.append({"phase": name, "target_position_root": target.tolist(),
                        "joint_position": (arm - JOINT_OFFSETS[:5]).tolist(),
                        "path_joint_positions": (path - JOINT_OFFSETS[:5]).tolist(),
                        "position_error_m": float(np.linalg.norm(position - target)),
                        "inclination_limit_violation": max(0., float(np.cos(np.pi / 4) - rotation[2, 2])) if index < 2 else None,
                        "maximum_penetration_m": depth, "sampled_path_maximum_penetration_m": maximum_depth,
                        "path_samples": len(samples), "accepted": bool(accepted), "attempts": len(solutions) if index < 2 else len(lift_solutions)})
        previous = arm
    return {"environment": environment["environment"], "targets": targets,
            "accepted": all(item["accepted"] for item in targets),
            "approach_grasp_rotation_difference": float(np.linalg.norm(pose(arms[0])[1] - pose(arms[1])[1]))}
