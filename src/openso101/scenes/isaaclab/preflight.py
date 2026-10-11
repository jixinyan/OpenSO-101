import math
from pathlib import Path

from .usd import verify_compilation


def validate_runtime_parameters(*, num_envs: int, steps: int, resets: int,
                                timeout_seconds: float | None = None) -> None:
    for name, value in (("num_envs", num_envs), ("steps", steps), ("resets", resets)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if timeout_seconds is not None:
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds 必须为正数有限数值")


def validate_scene_output(source: Path, output: Path) -> Path:
    source = source.resolve()
    if output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("场景输出目录与输入目录必须独立")
    if output.exists():
        raise FileExistsError(output)
    for parent in output.parents:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(parent)
    return output


def validate_runtime_scene(scene: Path) -> dict:
    compilation = verify_compilation(scene)
    if "environment.usda" not in compilation["files"]:
        raise ValueError("原生场景需要通过 SHA256 检查的 environment.usda")
    return compilation


def validate_runtime_output(scene: Path, output: Path) -> Path:
    output = validate_scene_output(scene, output)
    trace = output.with_suffix(".hdf5")
    if trace == output:
        raise ValueError("场景报告路径与 HDF5 轨迹路径必须独立")
    validate_scene_output(scene, trace)
    return output
