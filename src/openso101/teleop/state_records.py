def validate_replay_task_state(recording, task: str) -> None:
    source_task = recording.attrs["env_id"]
    source_task = source_task.decode() if isinstance(source_task, bytes) else str(source_task)
    if task != source_task:
        raise ValueError("回放任务需要与记录的 env_id 一致")
    if task == "OpenSO101-Stack-v0":
        objects = ("cube_top", "cube_bottom")
        fields = ("cube_top_was_lifted", "task_episode_step", "environment_origin")
    elif task in ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"):
        objects, fields = ("object",), ("environment_origin",)
    else:
        return
    for name in (*fields, *(f"{name}_root_state" for name in objects)):
        if f"sim/{name}" not in recording:
            raise ValueError(f"回放任务需要保存的场景状态: sim/{name}")
    for frame_index in range(len(recording["action"])):
        origin = recording["sim/environment_origin"][frame_index]
        for name in objects:
            replay_root_state(recording[f"sim/{name}_root_state"][frame_index], origin, origin)


def replay_root_state(values, source_origin, target_origin):
    import numpy as np

    state = np.asarray(values)
    source, target = np.asarray(source_origin), np.asarray(target_origin)
    if state.shape != (13,) or state.dtype.kind != "f" or not np.isfinite(state).all():
        raise ValueError("回放物体需要十三个有限的浮点状态值")
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (source, target)):
        raise ValueError("回放环境原点需要三个有限坐标")
    if not np.isclose(np.linalg.norm(state[3:7]), 1, atol=0.00001, rtol=0):
        raise ValueError("回放物体需要单位 quaternion")
    restored = state.copy()
    restored[:3] += np.asarray(target - source, dtype=state.dtype)
    return restored
