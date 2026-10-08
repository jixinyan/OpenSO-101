import math


def control_rate_fps(physics_dt: float, decimation: int) -> int:
    if not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("physics_dt 必须为有限正数")
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
        raise ValueError("decimation 必须为正整数")
    rate = 1.0 / (physics_dt * decimation)
    fps = round(rate)
    if fps < 1 or not math.isclose(rate, fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError("当前 LeRobot 录制需要整数控制 FPS")
    return fps


def _env_control_rate_fps(unwrapped_env) -> int:
    return control_rate_fps(float(unwrapped_env.cfg.sim.dt), unwrapped_env.cfg.decimation)


def _resolve_record_fps(unwrapped_env, requested_fps: int | None) -> int:
    fps = _env_control_rate_fps(unwrapped_env)
    if requested_fps is not None and requested_fps != fps:
        raise ValueError(f"--fps 必须与环境控制频率一致: {fps} Hz")
    print(f"[INFO]: 录制 FPS 使用环境控制频率: {fps} Hz。")
    return fps
