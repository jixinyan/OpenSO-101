import argparse
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys

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
    if sys.platform == "linux" and os.environ.get("OPENSO101_GPU_NAMESPACE") == "1":
        physical_gpu = int(os.environ["OPENSO101_PHYSICAL_GPU"])
        scope.validate_allocation([physical_gpu])
        from .gpu_guard import verify_guard

        verify_guard(physical_gpu)
        if list(Path("/dev").glob("nvidia[0-9]*")) != [Path(f"/dev/nvidia{physical_gpu}")]:
            raise RuntimeError("GPU 设备目录必须只包含指定物理设备")
        renderer_gpu = 0
    else:
        physical_gpu = _requested_gpu(scope)
        if sys.platform == "linux":
            guard_pid = os.environ.get("OPENSO101_GPU_GUARD_PID")
            if guard_pid is None:
                from .gpu_guard import automatic_report, run_guarded

                repo = Path(__file__).resolve().parents[3]
                raise SystemExit(run_guarded(sys.orig_argv, physical_gpu, repo, automatic_report(repo)))
            from .gpu_guard import verify_guard

            verify_guard(physical_gpu)
            _isolate_gpu(physical_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        renderer_gpu = physical_gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # CUDA 使用可见设备编号，renderer 使用设备目录中的编号。
    settings = {
        "--/renderer/activeGpu": str(renderer_gpu),
        "--/renderer/multiGpu/enabled": "false",
        "--/renderer/multiGpu/autoEnable": "false",
        "--/renderer/multiGpu/maxGpuCount": "1",
        "--/renderer/gpuEnumeration/glInterop/enabled": "false",
        "--/physics/cudaDevice": "0",
    }
    parser = argparse.ArgumentParser(add_help=False)
    for name, expected in settings.items():
        parser.add_argument(name, choices=(expected,), default=expected)
    parser.parse_known_args()
    for name, expected in settings.items():
        argument = f"{name}={expected}"
        if argument not in sys.argv:
            sys.argv.append(argument)
    return physical_gpu


def _requested_gpu(scope):
    value = os.environ.get("CUDA_VISIBLE_DEVICES", str(scope.default_gpu))
    devices = [int(item) for item in next(csv.reader([value]))]
    scope.validate_allocation(devices)
    if len(devices) != 1:
        raise ValueError("单个 Isaac 进程需要指定一个物理 GPU")
    return devices[0]


def _isolate_gpu(physical_gpu):
    executable = Path(os.environ.get("OPENSO101_BWRAP_PATH",
                      shutil.which("bwrap") or Path(sys.prefix).parent / "openso101-gpu-scope/usr/bin/bwrap"))
    if not executable.is_file():
        raise FileNotFoundError(f"Linux GPU 设备隔离需要 bubblewrap：{executable}")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    devices = {int(row[0]): row[1].strip() for row in csv.reader(io.StringIO(inventory))}
    uuid = devices[physical_gpu]
    command = [str(executable), "--unshare-user", "--bind", "/", "/", "--dev", "/dev"]
    for name in (f"nvidia{physical_gpu}", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset"):
        path = Path("/dev") / name
        if not path.exists():
            raise FileNotFoundError(path)
        command.extend(["--dev-bind", str(path), str(path)])
    command.extend(["--chdir", str(Path.cwd()), "--", *sys.orig_argv])
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": uuid, "NVIDIA_VISIBLE_DEVICES": uuid,
                               "OPENSO101_GPU_NAMESPACE": "1", "OPENSO101_PHYSICAL_GPU": str(physical_gpu)}
    os.execve(executable, command, environment)
