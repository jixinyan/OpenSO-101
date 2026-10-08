# GPU 使用与停止控制

项目最多使用单张物理 GPU 2。启动要求设备没有任何用户的 compute 或 graphics 进程，GPU utilization 为零，已用显存不超过 128 MiB。设备已有作业时，启动入口保存 `device_busy` 报告并返回 exit code 75，工作程序保持未启动状态。

`scripts/run_native_python.sh` 使用 `scripts/run_idle_gpu.py` 检查实际设备清单。直接调用 Isaac 的项目入口同样需要启动检查父进程。启动记录包含物理设备、UUID、运行用户、worker PID、时间与进程状态；记录保存在 `outputs/rl_progress/gpu_guard/`。

运行期间检查 compute 与 graphics 进程。如果出现其他项目的进程，包括 haomin 的进程，检查程序发送 SIGTERM 终止本项目的独立进程组，并等待子进程终止。终止等待时间超过 30 秒时清理仍在运行的本项目子进程。其他项目进程保持独立运行。

`scripts/audit_project_gpu.py --stop --output <新报告>` 根据当前用户、项目目录和程序入口识别实际 GPU 作业，记录终止结果并重新检查设备。仅查询时省略 `--stop`。

CPU 检查使用 `scripts/run_cpu_python.sh native <程序入口>` 或 `scripts/run_cpu_python.sh mujoco <程序入口>`。这两个入口禁止 CUDA，并配置已安装的 FFmpeg 与 OSMesa 动态库。MuJoCo 与 Isaac 使用独立 Python 环境。

本次实际占用检查已确认 GPU 2 的现有作业阻止 OpenSO-101 启动。真实 CPU 子进程检查覆盖进程组停止与父进程退出后的子进程清理。运行期间的 GPU 竞争检查继续等待实际设备验收。
