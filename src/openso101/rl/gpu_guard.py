import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import psutil
from filelock import FileLock


def gpu_inventory():
    root = ElementTree.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
    devices = []
    for index, gpu in enumerate(root.findall("gpu")):
        processes = []
        for item in gpu.findall("processes/process_info"):
            pid = int(item.findtext("pid"))
            processes.append({"pid": pid, "owner": psutil.Process(pid).username(),
                              "type": item.findtext("type"), "used_memory": item.findtext("used_memory")})
        devices.append({"index": index, "uuid": gpu.findtext("uuid"),
                        "memory_used_mib": float(gpu.findtext("fb_memory_usage/used").removesuffix(" MiB")),
                        "utilization_percent": float(gpu.findtext("utilization/gpu_util").removesuffix(" %")),
                        "processes": processes})
    if not devices:
        raise RuntimeError("没有取得实际 GPU 清单")
    return devices


def idle_device(device):
    return not device["processes"] and device["utilization_percent"] == 0 and device["memory_used_mib"] <= 128


def owned_processes(process):
    if process.poll() is not None:
        return set()
    root = psutil.Process(process.pid)
    return {item.pid for item in (root, *root.children(recursive=True))}


def terminate_group(process, *, timeout=30, tracked=()):
    children = {item.pid: item for item in tracked if item.pid != process.pid}
    if process.poll() is None:
        root = psutil.Process(process.pid)
        if os.getpgid(process.pid) != process.pid or root.uids().real != os.getuid():
            raise RuntimeError("终止控制需要属于当前用户的独立进程组")
        children.update({item.pid: item for item in root.children(recursive=True)})
        os.killpg(process.pid, signal.SIGTERM)
    else:
        _, active = psutil.wait_procs(list(children.values()), timeout=0)
        for child in active:
            if child.uids().real != os.getuid():
                raise RuntimeError("子进程终止需要匹配当前用户")
            child.terminate()
    deadline = time.monotonic() + timeout
    _, alive = psutil.wait_procs(list(children.values()), timeout=0)
    while (process.poll() is None or alive) and time.monotonic() < deadline:
        time.sleep(.1)
        _, alive = psutil.wait_procs(alive, timeout=0)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    for child in alive:
        if child.uids().real != os.getuid():
            raise RuntimeError("子进程终止需要匹配当前用户")
        child.kill()
    _, remaining = psutil.wait_procs(alive, timeout=5)
    if remaining:
        raise RuntimeError("本项目的子进程尚未终止")
    return process.wait(timeout=5)


def verify_guard(gpu):
    ready_fd = os.environ.pop("OPENSO101_GPU_GUARD_READY_FD", None)
    if ready_fd is not None:
        descriptor = int(ready_fd)
        if os.read(descriptor, 1) != b"1":
            raise RuntimeError("GPU 启动检查没有完成进程登记")
        os.close(descriptor)
    guard_pid = int(os.environ["OPENSO101_GPU_GUARD_PID"])
    report = Path(os.environ["OPENSO101_GPU_GUARD_REPORT"])
    root = psutil.Process()
    parents = {item.pid: item for item in root.parents()}
    if guard_pid not in parents or parents[guard_pid].uids().real != os.getuid():
        raise RuntimeError("GPU 作业需要属于当前用户的启动检查父进程")
    receipt = json.loads(report.read_text())
    if (receipt["guard_pid"] != guard_pid or receipt["gpu"] != gpu or receipt["status"] != "running"
            or receipt["worker_pid"] not in {root.pid, *parents} or not receipt["gpu_job_started"]):
        raise RuntimeError("GPU 作业与实际启动记录不一致")


def run_guarded(command, gpu, repo, report):
    from .gpu_scope import gpu_scope

    gpu_scope().validate_allocation([gpu])
    repo = repo.resolve()
    report = report.resolve()
    if not report.is_relative_to(repo / "outputs") or report.exists():
        raise ValueError("GPU 检查需要位于 outputs 的新报告文件")
    report.parent.mkdir(parents=True, exist_ok=True)
    lock_root = repo / "outputs/tmp/gpu_guard"
    lock_root.mkdir(parents=True, exist_ok=True)
    receipt = {"created_at": datetime.now(UTC).isoformat(), "gpu": gpu, "command": command,
               "guard_pid": os.getpid(), "status": "checking_idle_device", "gpu_job_started": False,
               "protected_users": ["haomin"], "requires_exclusive_device": True}

    def save():
        report.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    save()
    process = None
    tracked = {}
    descriptors = set()
    requested_signal = []

    def stop_requested(signum, _frame):
        requested_signal.append(signum)

    handlers = {signum: signal.signal(signum, stop_requested) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with FileLock(lock_root / f"gpu_{gpu}.lock", timeout=0):
            device = next(item for item in gpu_inventory() if item["index"] == gpu)
            receipt["initial_device"] = device
            if not idle_device(device):
                receipt.update(status="device_busy", completed_at=datetime.now(UTC).isoformat(), exit_code=75)
                save()
                print(f"GPU {gpu} 已有作业或设备活动，未启动本项目进程。报告：{report}", flush=True)
                return 75
            if requested_signal:
                receipt.update(status="stop_requested", signal=requested_signal[0],
                               completed_at=datetime.now(UTC).isoformat(), exit_code=128 + requested_signal[0])
                save()
                return receipt["exit_code"]
            ready_read, ready_write = os.pipe()
            descriptors.update((ready_read, ready_write))
            environment = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu),
                                         "OPENSO101_GPU_GUARD_PID": str(os.getpid()),
                                         "OPENSO101_GPU_GUARD_REPORT": str(report),
                                         "OPENSO101_GPU_GUARD_READY_FD": str(ready_read)}
            entry = [sys.executable, "-u", str(repo / "scripts/gpu_worker_entry.py"), *command]
            process = subprocess.Popen(entry, cwd=repo, env=environment, start_new_session=True,
                                       pass_fds=(ready_read,))
            os.close(ready_read)
            descriptors.remove(ready_read)
            receipt.update(status="running", gpu_job_started=True, worker_pid=process.pid)
            save()
            os.write(ready_write, b"1")
            os.close(ready_write)
            descriptors.remove(ready_write)
            while process.poll() is None and not requested_signal:
                owned = owned_processes(process)
                tracked.update({pid: psutil.Process(pid) for pid in owned})
                device = next(item for item in gpu_inventory() if item["index"] == gpu)
                # 再次读取进程树，包含 GPU 查询期间启动的子进程。
                owned.update(owned_processes(process))
                foreign = [item for item in device["processes"] if item["pid"] not in owned]
                if foreign:
                    receipt.update(status="foreign_job_detected", conflicting_processes=foreign,
                                   conflict_detected_at=datetime.now(UTC).isoformat())
                    save()
                    terminate_group(process, tracked=tracked.values())
                    break
                time.sleep(.5)
            if requested_signal:
                receipt.update(status="stop_requested", signal=requested_signal[0])
                save()
                terminate_group(process, tracked=tracked.values())
            terminate_group(process, tracked=tracked.values())
            if receipt["status"] == "running":
                receipt["status"] = "completed" if process.returncode == 0 else "worker_failed"
            result = 75 if receipt["status"] == "foreign_job_detected" else (
                process.returncode if process.returncode >= 0 else 128 - process.returncode)
            receipt.update(exit_code=result, worker_exit_code=process.returncode,
                           completed_at=datetime.now(UTC).isoformat())
            save()
            return result
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
        if process is not None:
            running = process.poll() is None
            terminate_group(process, tracked=tracked.values())
        else:
            running = False
        if running:
            receipt.update(status="guard_terminated_worker", worker_exit_code=process.returncode,
                           completed_at=datetime.now(UTC).isoformat())
            save()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def automatic_report(repo):
    return repo / "outputs/rl_progress/gpu_guard" / f"{time.time_ns()}_{os.getpid()}.json"


def _cuda_device(device):
    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and requested.index not in (None, 0):
        raise ValueError("单张 GPU 的 Torch device 需要使用 cuda:0")
    return requested


def _cuda_receipt():
    from .gpu_scope import gpu_scope

    if "OPENSO101_GPU_GUARD_PID" not in os.environ or "OPENSO101_GPU_GUARD_REPORT" not in os.environ:
        raise ValueError("CUDA 推理需要设备检查父进程")
    receipt = json.loads(Path(os.environ["OPENSO101_GPU_GUARD_REPORT"]).read_text())
    gpu_scope().validate_allocation([receipt["gpu"]])
    verify_guard(receipt["gpu"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (str(receipt["gpu"]), receipt["initial_device"]["uuid"]):
        raise ValueError("CUDA 可见设备与实际启动记录不一致")
    return receipt


def launch_cuda_command(device):
    requested = _cuda_device(device)
    if requested.type != "cuda" or sys.platform != "linux":
        return
    if "OPENSO101_GPU_GUARD_PID" not in os.environ:
        from .gpu_scope import _requested_gpu, gpu_scope

        gpu = _requested_gpu(gpu_scope())
        repo = Path(__file__).resolve().parents[3]
        raise SystemExit(run_guarded(sys.orig_argv, gpu, repo, automatic_report(repo)))
    receipt = _cuda_receipt()
    import torch

    if torch.cuda.is_initialized() and os.environ["CUDA_VISIBLE_DEVICES"] != receipt["initial_device"]["uuid"]:
        raise ValueError("CUDA 初始化前需要使用启动记录的设备 UUID")
    os.environ["CUDA_VISIBLE_DEVICES"] = receipt["initial_device"]["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def verify_cuda_inference(device):
    requested = _cuda_device(device)
    if requested.type != "cuda" or sys.platform != "linux":
        return
    receipt = _cuda_receipt()
    import torch

    if torch.cuda.is_initialized() and os.environ["CUDA_VISIBLE_DEVICES"] != receipt["initial_device"]["uuid"]:
        raise ValueError("CUDA 初始化前需要使用启动记录的设备 UUID")
    os.environ["CUDA_VISIBLE_DEVICES"] = receipt["initial_device"]["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if torch.cuda.device_count() != 1:
        raise ValueError("CUDA 推理需要只看到启动记录指定的单张 GPU")
