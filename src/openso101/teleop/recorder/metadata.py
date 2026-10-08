from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from openso101.scenes.models import Model
from openso101.teleop.simulation import RecordedSimulation
from openso101.teleop.timing import control_rate_fps


RECORDING_METADATA_FILE = "meta/openso101_recording.json"


class RecordingEpisode(Model):
    episode_index: int = Field(ge=0, strict=True)
    frames: int = Field(gt=0, strict=True)
    task: str = Field(min_length=1)
    success: bool = Field(strict=True)


class RecordingMetadata(Model):
    schema_version: Literal[1] = 1
    fps: int = Field(gt=0, strict=True)
    simulation: RecordedSimulation | None = None
    scene_relative_path: str | None = None
    episodes: tuple[RecordingEpisode, ...] = ()

    @model_validator(mode="after")
    def recording_source(self):
        if self.simulation is not None and control_rate_fps(
                self.simulation.physics_dt, self.simulation.decimation) != self.fps:
            raise ValueError("LeRobot 录制物理周期与 FPS 不一致")
        scene_digest = self.simulation.scene_sha256 if self.simulation is not None else None
        if (scene_digest is None) != (self.scene_relative_path is None):
            raise ValueError("LeRobot 场景来源需要同时保存 SHA256 与相对路径")
        if self.scene_relative_path is not None:
            path = Path(self.scene_relative_path)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != f"scenes/{scene_digest}":
                raise ValueError("LeRobot 录制场景需要对应 SHA256 的相对路径")
        if [item.episode_index for item in self.episodes] != list(range(len(self.episodes))):
            raise ValueError("LeRobot 录制 episode_index 必须连续")
        return self


def load_recording_metadata(root: Path) -> RecordingMetadata | None:
    path = root / RECORDING_METADATA_FILE
    if not path.is_file():
        return None
    metadata = RecordingMetadata.model_validate_json(path.read_text())
    if metadata.scene_relative_path is not None:
        from openso101.scenes.isaaclab.usd import verify_compilation

        scene = (root / metadata.scene_relative_path).resolve()
        if not scene.is_relative_to(root.resolve()):
            raise ValueError("LeRobot 录制场景需要位于数据集目录")
        if verify_compilation(scene)["scene_sha256"] != metadata.simulation.scene_sha256:
            raise ValueError("LeRobot 录制场景与来源 SHA256 不一致")
    return metadata


def validate_recording_metadata(root: Path, fps: int, episodes) -> None:
    metadata = load_recording_metadata(root)
    if metadata is None:
        return
    if metadata.fps != fps or len(metadata.episodes) != len(episodes):
        raise ValueError("LeRobot 录制来源与实际 FPS 或 episode 数量不一致")
    for recorded, actual in zip(metadata.episodes, episodes):
        if (recorded.episode_index != actual["episode_index"] or recorded.frames != actual["length"]
                or actual["tasks"] != [recorded.task]):
            raise ValueError("LeRobot 录制来源与实际 episode 帧数或任务不一致")
