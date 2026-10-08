import math
from typing import Literal

from pydantic import Field, model_validator

from openso101.scenes.models import Digest, Model
from openso101.teleop.timing import control_rate_fps


class RecordedSimulation(Model):
    env_id: str = Field(min_length=1)
    task_profile: Literal["teleop", "default", "grasp_v2", "grasp_v3", "grasp_v4"]
    physics_dt: float = Field(gt=0)
    decimation: int = Field(gt=0, strict=True)
    environment_mode: Literal["nominal", "randomized"] | None = None
    reward_discount: float | None = Field(default=None, gt=0, le=1)
    scene_sha256: Digest | None = None

    @model_validator(mode="after")
    def native_profile(self):
        control_rate_fps(self.physics_dt, self.decimation)
        if self.task_profile != "teleop" and self.environment_mode is None:
            raise ValueError("策略采集的仿真设置需要 environment_mode")
        if self.task_profile.startswith("grasp_"):
            if self.env_id not in ("OpenSO101-Lift-v0", "OpenSO101-PickPlace-v0"):
                raise ValueError("grasp 仿真设置需要 Lift 或 PickPlace")
            if self.environment_mode is None or self.scene_sha256 is not None:
                raise ValueError("grasp 仿真设置需要 environment_mode 和标准任务场景")
        if self.task_profile in ("grasp_v3", "grasp_v4") and self.reward_discount is None:
            raise ValueError("grasp_v3 和 grasp_v4 仿真设置需要 reward_discount")
        return self


def recorded_simulation(attrs) -> RecordedSimulation | None:
    if "physics_dt" not in attrs:
        if "task_profile" in attrs:
            raise ValueError("记录了 task_profile 的来源需要 physics_dt")
        return None
    physics_dt, fps = float(attrs["physics_dt"]), float(attrs["fps"])
    if not math.isfinite(physics_dt) or physics_dt <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("来源 physics_dt 与 FPS 需要有限正数")
    decimation = round(1 / (physics_dt * fps))
    if control_rate_fps(physics_dt, decimation) != fps:
        raise ValueError("来源物理周期、decimation 与 FPS 不一致")
    if "decimation" in attrs and attrs["decimation"] != decimation:
        raise ValueError("来源保存的 decimation 与物理周期不一致")
    return RecordedSimulation(
        env_id=str(attrs["env_id"]), task_profile=str(attrs.get("task_profile", "teleop")),
        physics_dt=physics_dt, decimation=decimation,
        environment_mode=attrs.get("environment_mode"), reward_discount=attrs.get("reward_discount"),
        scene_sha256=attrs.get("scene_sha256"),
    )


def runtime_simulation(runtime, task: str, scene_metadata=None, *, task_profile="teleop") -> RecordedSimulation:
    return RecordedSimulation(env_id=task, task_profile=task_profile, physics_dt=float(runtime.physics_dt),
                              decimation=int(runtime.cfg.decimation),
                              environment_mode=getattr(runtime.cfg, "environment_mode", None),
                              reward_discount=(runtime.cfg.reward_discount
                                               if task_profile in ("grasp_v3", "grasp_v4") else None),
                              scene_sha256=(scene_metadata or {}).get("scene_sha256"))
