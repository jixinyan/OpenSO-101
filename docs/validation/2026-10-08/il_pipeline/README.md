# CPU、LeRobot 导出与打包验证

被检查的代码为 `d367309b9f1fb5e4496822178f40f7c7298260f9`。`verification.json` 保存报告 SHA256；`suite.json` 保存源码、输入文件、参数和阶段结果。

统一 CPU 流程通过 12 个阶段。GitHub CI 通过 49 项几何检查和 55 项接口检查。源码检查覆盖 305 个 Python 文件、183 个模块、789 处项目 import 和 23 个 Shell 文件；七个命令组的实际帮助入口通过检查。Ruff 未定义名称检查结果为空列表。实际 wheel 在 CI 与 `jd_B300` 分别通过文件 SHA256、metadata、入口和 ZIP CRC 验证。

LeRobot 同步与异步导出分别读取实际 PickPlace HDF5。每种模式完整检查 328 帧、两台相机、动作、关节状态和时间戳。动作与关节状态转换误差均为零，时间戳最大误差为 `2.2888183615066282e-7` 秒。wrist 和 overhead 视频编码的最大平均绝对误差分别为 `0.0087176552` 和 `0.0059498702`。源文件 SHA256 保持一致；导出工作线程全部关闭；无可导出 episode 的请求保留已有数据文件及其 SHA256。

数据导出位于 `il/datasets/export.py`，相机与关节观测处理位于 `il/observations.py`，Isaac 启动与资源关闭位于 `il/runtime.py`，录制按键位于 `teleop/devices/recording_keys.py`。IL 和环境命令的无效参数通过实际子进程检查，在 Isaac 启动前终止。环境与应用的原生关闭路径继续等待 GPU 运行验收。

已有键盘 IK、视频标定、场景批量生成、LeRobot 读取、连续动作分析、optimizer 更新和 MuJoCo 策略运行完成 CPU 检查。MuJoCo 的四个 episode 共执行 20,000 个物理步骤，速度和力矩限制通过，任务成功为 0/4。最终 PPO 的既有独立任务评估为 0/100。

此次 `gpu_tests_started`、`native_physics_verified`、`full_v2_verified` 均为 `false`。人工键盘成功采集、成功 RL 策略 sim2sim、视觉 student 独立任务与真实视频恢复继续等待对应验收。模型、日志和原始数据全部保留。
