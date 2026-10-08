import os
import sys

from openso101.rl.gpu_guard import verify_guard


verify_guard(int(os.environ["CUDA_VISIBLE_DEVICES"]))
command = sys.argv[1:]
if not command:
    raise ValueError("GPU 工作进程需要明确的程序入口")
os.execvpe(command[0], command, os.environ)
