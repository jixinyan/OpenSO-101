import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from openso101.rl.gpu_guard import owned_processes, terminate_group


def start_worker(pid_file, *, session):
    script = Path(__file__).parent / "fixtures/process_group.py"
    process = subprocess.Popen([sys.executable, str(script), "--pid-file", str(pid_file)], start_new_session=session)
    deadline = time.monotonic() + 10
    while not pid_file.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("实际子进程没有启动")
        time.sleep(.01)
    return process, int(pid_file.read_text())


def test_actual_process_group_termination(tmp_path):
    process, child = start_worker(tmp_path / "child.pid", session=True)
    try:
        assert {process.pid, child}.issubset(owned_processes(process))
        assert terminate_group(process, timeout=2) == -signal.SIGTERM
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE:
            if time.monotonic() > deadline:
                raise RuntimeError("实际进程组仍有运行中的子进程")
            time.sleep(.01)
        assert owned_processes(process) == set()
    finally:
        if process.poll() is None:
            terminate_group(process, timeout=2)


def test_shared_process_group_is_rejected(tmp_path):
    process, child = start_worker(tmp_path / "shared_child.pid", session=False)
    try:
        assert os.getpgid(process.pid) == os.getpgid(os.getpid())
        with pytest.raises(RuntimeError, match="独立进程组"):
            terminate_group(process, timeout=2)
        assert process.poll() is None and psutil.Process(child).is_running()
    finally:
        psutil.Process(child).terminate()
        process.terminate()
        process.wait(timeout=5)


def test_actual_child_cleanup_after_parent_exit(tmp_path):
    pid_file = tmp_path / "orphan.pid"
    script = Path(__file__).parent / "fixtures/process_group.py"
    process = subprocess.Popen([sys.executable, str(script), "--parent-exit", "--pid-file", str(pid_file)],
                               start_new_session=True)
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while not pid_file.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("实际子进程没有写入 identifier")
        time.sleep(.01)
    child = psutil.Process(int(pid_file.read_text()))
    try:
        assert terminate_group(process, timeout=2, tracked=(child,)) == 0
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()
