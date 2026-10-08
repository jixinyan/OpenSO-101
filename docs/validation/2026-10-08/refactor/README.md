# CPU 验证与功能目录

源码 commit 为 `c5daa915a7b4ec1bef297a9769b6bcb53c399949`。本目录保存 `jd_B300` 的完整 CPU 验证、GitHub CI、实际 wheel 检查和逐报告 SHA256。[verification.json](verification.json) 记录汇总字段，[suite.json](suite.json) 保存逐阶段命令、时间、输入与源码 SHA256。

| 检查范围 | 实际结果 |
|---|---|
| 统一 CPU 流程 | 11 个阶段全部通过 |
| GitHub CPU 检查 | 49 项几何和进程检查、11 项 PTY、控制和配置检查通过 |
| 源码和命令 | 294 个 Python 文件、176 个模块、769 个项目 import、23 个 Shell 文件、7 个命令组 |
| 实际 wheel | 全部 176 个模块与当前源码逐文件 SHA256 相同，metadata 与命令入口通过 |
| 键盘 IK | 64 个实际录制姿态，512 次官方 MJCF FK 检查 |
| 视频位置恢复 | MuJoCo CPU OSMesa 视频，4 个独立验证点最大位置误差 0.82 mm |
| 场景与数据 | 10 个已有 bundle 完成静态检查，328 帧双相机 LeRobot 数据完成读取 |
| 保存模型与 optimizer | 连续监督分析和两次 CPU 更新通过，来源归一化保持不变 |
| MuJoCo 策略 | 4 个 episode、20,000 个物理步骤，实际速度和力矩限制通过，任务成功 0/4 |
| GPU 进程 | 检查时本项目没有 GPU 作业，其他项目进程继续运行 |

MuJoCo 使用已有 `v4_seed44_99_b49353f_portable` 策略。其任务结果见 [策略报告](suite/mujoco_policy/report.json)。本次 CPU 检查没有新增 RL transitions，没有执行原生 Isaac GPU 物理检查，`full_v2_verified` 为 `false`。人工键盘成功采集、RL 收敛、视觉 student 任务和真机验收继续等待对应记录。

资产管理、模型生成、布局、场景版本、Isaac Lab、键盘设备与录制分别位于独立的功能目录。文件位置与执行入口见 [代码目录](../../../guides/code-map.md)。

原始模型、optimizer、轨迹、日志和视频继续保存于 canonical 仓库的 `outputs/rl_progress/`。本目录中的 JSON 和 XML 来自实际运行报告，复制时检查逐文件 SHA256。
