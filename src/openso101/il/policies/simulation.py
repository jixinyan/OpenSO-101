import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from openso101.scenes.models import Digest, Model, file_digest
from openso101.teleop.so101_mapping import LEROBOT_SO101_ACTION_NAMES


SIMULATION_SETTINGS_FILE = "openso101_simulation.json"


class SimulationSettings(Model):
    schema_version: Literal[1] = 1
    fps: int = Field(gt=0, strict=True)
    camera_sizes: dict[str, tuple[int, int]]
    joint_names: tuple[str, ...] = LEROBOT_SO101_ACTION_NAMES
    action_units: Literal["motor_units"] = "motor_units"
    dataset_info_sha256: Digest | None = None

    @model_validator(mode="after")
    def supported_inputs(self):
        if self.joint_names != LEROBOT_SO101_ACTION_NAMES:
            raise ValueError("IL 仿真设置需要 SO-101 的六个 LeRobot 关节名称与顺序")
        if set(self.camera_sizes) != {"wrist_camera", "overhead_camera"}:
            raise ValueError("IL 仿真设置需要 wrist_camera 和 overhead_camera")
        if any(type(value) is not int or value < 16 for shape in self.camera_sizes.values() for value in shape):
            raise ValueError("IL 仿真相机的 height 和 width 必须为至少 16 的整数")
        return self


def _camera_sizes(configuration: dict) -> dict:
    cameras = {}
    for name, feature in configuration["input_features"].items():
        if feature["type"] == "VISUAL":
            shape = feature["shape"]
            if len(shape) != 3 or shape[0] != 3 or not name.startswith("observation.images."):
                raise ValueError("IL 模型相机需要 [3, height, width] observation.images")
            cameras[name.removeprefix("observation.images.")] = tuple(shape[1:])
    if configuration["input_features"]["observation.state"]["shape"] != [6] or configuration["output_features"]["action"]["shape"] != [6]:
        raise ValueError("IL 仿真需要六个 observation.state 和 action 关节值")
    return cameras


def save_simulation_settings(checkpoint: Path, dataset) -> SimulationSettings:
    configuration = json.loads((checkpoint / "config.json").read_text())
    for name in ("action", "observation.state"):
        if tuple(dataset.meta.features[name]["names"]) != LEROBOT_SO101_ACTION_NAMES:
            raise ValueError(f"IL 数据集的关节名称与顺序不一致: {name}")
    settings = SimulationSettings(fps=dataset.meta.fps, camera_sizes=_camera_sizes(configuration),
                                  dataset_info_sha256=file_digest(Path(dataset.root) / "meta/info.json"))
    with (checkpoint / SIMULATION_SETTINGS_FILE).open("x") as stream:
        stream.write(settings.model_dump_json(indent=2) + "\n")
    return settings


def load_simulation_settings(checkpoint: Path, control_fps: int | None = None) -> SimulationSettings | None:
    if (checkpoint / "student.json").is_file():
        if control_fps is not None:
            raise ValueError("student 的仿真评估使用 rl student-eval")
        return None
    configuration = json.loads((checkpoint / "config.json").read_text())
    path = checkpoint / SIMULATION_SETTINGS_FILE
    if path.is_file():
        settings = SimulationSettings.model_validate_json(path.read_text())
        if control_fps is not None and control_fps != settings.fps:
            raise ValueError("control_fps 与模型保存的采集频率不一致")
        if settings.camera_sizes != _camera_sizes(configuration):
            raise ValueError("模型保存的相机尺寸与 input_features 不一致")
        return settings
    if control_fps is None:
        raise ValueError("模型需要 openso101_simulation.json；已有模型需要明确提供 --control-fps")
    return SimulationSettings(fps=control_fps, camera_sizes=_camera_sizes(configuration))


def apply_simulation_settings(cfg, settings: SimulationSettings | None) -> None:
    if settings is None:
        return
    cfg.sim.dt = 1 / (settings.fps * cfg.decimation)
    cfg.sim.render_interval = cfg.decimation
    for name, (height, width) in settings.camera_sizes.items():
        camera = getattr(cfg.scene, name)
        camera.height, camera.width = height, width


def attach_simulation_settings(output: Path, source: Path) -> list[str]:
    settings = SimulationSettings.model_validate_json(source.read_text())
    copied = []
    if not output.exists():
        return copied
    checkpoints = sorted({path.parent.resolve() for path in output.rglob("config.json")
                          if path.parent.name == "pretrained_model"})
    for checkpoint in checkpoints:
        configuration = json.loads((checkpoint / "config.json").read_text())
        if settings.camera_sizes != _camera_sizes(configuration):
            raise ValueError("训练 checkpoint 的 input_features 与保存的仿真设置不一致")
        destination = checkpoint / SIMULATION_SETTINGS_FILE
        if destination.exists():
            if SimulationSettings.model_validate_json(destination.read_text()) != settings:
                raise ValueError("训练 checkpoint 中的仿真设置已经存在，并且内容不一致")
        else:
            with destination.open("x") as stream:
                stream.write(settings.model_dump_json(indent=2) + "\n")
        copied.append(str(destination))
    return copied
