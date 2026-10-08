import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from openso101.scenes.models import Digest, Model, file_digest
from openso101.teleop.so101_mapping import LEROBOT_SO101_ACTION_NAMES
from openso101.teleop.simulation import RecordedSimulation
from openso101.teleop.timing import control_rate_fps


SIMULATION_SETTINGS_FILE = "openso101_simulation.json"


class SimulationSettings(Model):
    schema_version: Literal[1] = 1
    fps: int = Field(gt=0, strict=True)
    camera_sizes: dict[str, tuple[int, int]]
    joint_names: tuple[str, ...] = LEROBOT_SO101_ACTION_NAMES
    action_units: Literal["motor_units"] = "motor_units"
    dataset_info_sha256: Digest | None = None
    dataset_export_sha256: Digest | None = None
    recorded_simulation: RecordedSimulation | None = None

    @model_validator(mode="after")
    def supported_inputs(self):
        if self.joint_names != LEROBOT_SO101_ACTION_NAMES:
            raise ValueError("IL 仿真设置需要 SO-101 的六个 LeRobot 关节名称与顺序")
        if set(self.camera_sizes) != {"wrist_camera", "overhead_camera"}:
            raise ValueError("IL 仿真设置需要 wrist_camera 和 overhead_camera")
        if any(type(value) is not int or value < 16 for shape in self.camera_sizes.values() for value in shape):
            raise ValueError("IL 仿真相机的 height 和 width 必须为至少 16 的整数")
        source = self.recorded_simulation
        if source is not None and control_rate_fps(source.physics_dt, source.decimation) != self.fps:
            raise ValueError("IL 来源物理周期与模型控制 FPS 不一致")
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
    simulation, export_digest = dataset_simulation(Path(dataset.root), dataset.meta.fps, dataset.num_episodes)
    settings = SimulationSettings(fps=dataset.meta.fps, camera_sizes=_camera_sizes(configuration),
                                  dataset_info_sha256=file_digest(Path(dataset.root) / "meta/info.json"),
                                  dataset_export_sha256=export_digest,
                                  recorded_simulation=simulation)
    with (checkpoint / SIMULATION_SETTINGS_FILE).open("x") as stream:
        stream.write(settings.model_dump_json(indent=2) + "\n")
    return settings


def dataset_simulation(root: Path, fps: int, episodes: int):
    path = root / "meta/openso101_export.json"
    if not path.is_file():
        return None, None
    export = json.loads(path.read_text())
    if export["schema_version"] != 1 or export["fps"] != fps or len(export["episodes"]) != episodes:
        raise ValueError("IL 导出来源与 LeRobot 的 schema_version、FPS 或 episode 数量不一致")
    sources = [RecordedSimulation.model_validate(item["simulation"]) if item.get("simulation") is not None
               else None for item in export["episodes"]]
    if not sources or any(source != sources[0] for source in sources):
        raise ValueError("IL 模型需要一致的来源物理参数、任务和场景")
    source = sources[0]
    if source is not None and (control_rate_fps(source.physics_dt, source.decimation) != fps or
            any(item["env_id"] != source.env_id for item in export["episodes"])):
        raise ValueError("IL 导出的仿真来源与 FPS 或任务名称不一致")
    return source, file_digest(path)


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
    if settings.recorded_simulation is not None:
        cfg.decimation = settings.recorded_simulation.decimation
        cfg.sim.dt = settings.recorded_simulation.physics_dt
    else:
        cfg.sim.dt = 1 / (settings.fps * cfg.decimation)
    cfg.sim.render_interval = cfg.decimation
    for name, (height, width) in settings.camera_sizes.items():
        camera = getattr(cfg.scene, name)
        camera.height, camera.width = height, width


def validate_simulation_request(settings, task: str, scene=None) -> None:
    source = settings.recorded_simulation if settings is not None else None
    if source is None:
        return
    if task != source.env_id:
        raise ValueError("IL 请求的任务与模型保存的采集任务不一致")
    if source.scene_sha256 is not None:
        from openso101.scenes.isaaclab.usd import verify_compilation

        if scene is None or verify_compilation(Path(scene).expanduser().resolve())["scene_sha256"] != source.scene_sha256:
            raise ValueError("IL 请求的场景与模型保存的采集场景不一致")
    elif scene is not None:
        raise ValueError("IL 标准任务模型需要对应的标准场景")


def configure_policy_simulation(cfg, settings, task: str, scene=None) -> None:
    validate_simulation_request(settings, task, scene)
    source = settings.recorded_simulation if settings is not None else None
    if source is not None and source.task_profile != "teleop":
        from isaaclab.envs.mdp import JointPositionActionCfg
        from openso101.robots import SO101_ARM_JOINT_NAMES, SO101_GRIPPER_JOINT_NAMES
        from openso101.tasks.shared.grasp_v3 import configure_grasp_v3, configure_environment_mode
        from openso101.tasks.shared.grasp_v4 import configure_grasp_v4
        from openso101.tasks.shared.grasp_profile import configure_grasp_profile

        profiles = {"grasp_v2": configure_grasp_profile, "grasp_v3": configure_grasp_v3,
                    "grasp_v4": configure_grasp_v4}
        if source.task_profile != "default":
            profiles[source.task_profile](cfg, task)
        cfg.configure_play(True)
        if scene is not None:
            cfg.configure_scene(scene)
        configure_environment_mode(cfg, source.environment_mode)
        if source.task_profile in ("grasp_v3", "grasp_v4") and (cfg.sim.dt != source.physics_dt or cfg.decimation != source.decimation):
            raise ValueError("IL 来源物理周期与保存的 grasp profile 不一致")
        if source.reward_discount is not None:
            cfg.reward_discount = source.reward_discount
        cfg.action_dr_enabled = False
        cfg.actions = {
            "arm_action": JointPositionActionCfg(asset_name="robot", joint_names=list(SO101_ARM_JOINT_NAMES),
                preserve_order=True, use_default_offset=False),
            "gripper_action": JointPositionActionCfg(asset_name="robot", joint_names=list(SO101_GRIPPER_JOINT_NAMES),
                preserve_order=True, use_default_offset=False, clip={".*": (0., .8)}),
        }
        if hasattr(cfg.observations.policy, "task_state"):
            cfg.observations.policy.task_state = None
        cfg.rewards, cfg.terminations, cfg.curriculum = None, None, None
    else:
        cfg.configure_action_mode("teleop")
    cfg.configure_cameras(True)
    if scene is not None and (source is None or source.task_profile == "teleop"):
        cfg.configure_scene(scene)
    apply_simulation_settings(cfg, settings)


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
