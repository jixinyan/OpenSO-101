import csv
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GpuScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    allowed_gpus: list[int] = Field(min_length=1)
    default_gpu: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_devices(self):
        if (min(self.allowed_gpus) < 0 or len(set(self.allowed_gpus)) != len(self.allowed_gpus)
                or self.default_gpu not in self.allowed_gpus):
            raise ValueError("GPU 范围需要唯一的非负编号，默认 GPU 必须包含在允许范围内")
        return self

    def validate_allocation(self, devices):
        if not devices or any(device not in self.allowed_gpus for device in devices):
            raise ValueError(f"请求的 GPU {devices} 超出允许范围 {self.allowed_gpus}")


def gpu_scope():
    path = Path(os.environ.get("OPENSO101_GPU_SCOPE", Path(__file__).resolve().parents[3] / "configs/runtime/gpu_scope.json"))
    return GpuScope.model_validate_json(path.read_text())


def configure_visible_gpu():
    scope = gpu_scope()
    value = os.environ.get("CUDA_VISIBLE_DEVICES", str(scope.default_gpu))
    devices = [int(item) for item in next(csv.reader([value]))]
    scope.validate_allocation(devices)
    if len(devices) != 1:
        raise ValueError("单个 Isaac 进程需要指定一个物理 GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    return devices[0]
