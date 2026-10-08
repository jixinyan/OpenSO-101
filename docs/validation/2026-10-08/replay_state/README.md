# CPU 状态、metadata 与视觉 student 验证

被检查的代码为 `e7b019dba87871fb2c5a716df45d19e284541463`。`verification.json` 保存全部报告的 SHA256；`suite.json` 保存源码、输入、参数与阶段结果。

统一 CPU 流程通过 14 个阶段。GitHub CI 通过 49 项几何检查和 65 项接口检查，合计 114 项。源码报告覆盖 310 个 Python 文件、186 个模块、798 处项目 import 和 23 个 Shell 文件；七个命令组的实际帮助入口通过检查。Ruff 未定义名称检查结果为空列表。CI 构建实际 sdist 与 wheel；`jd_B300` 核查同一 wheel 的源码、metadata、入口与 ZIP CRC，wheel SHA256 为 `3a83a3ca7d9bdc887671b1ee88295d5083d4a49296789f742e38b67566720977`。

`recorded_state` 读取实际 PickPlace HDF5 的全部 328 帧，对 23 个状态字段执行 CPU 转换，误差均为零。保存的 50 Hz、`physics_dt=0.001` 与推导的 `decimation=20` 相符。场景录制入口通过实际子进程检查，在启动 Isaac 前拒绝零步骤。原生机器人和场景状态恢复继续等待 GPU 运行验收。

LeRobot 同步与异步导出各检查 328 帧和两台相机，动作与关节状态转换误差均为零。metadata 的连续 episode 范围、实际 Parquet 帧数、六个关节顺序、相机 FPS、有限统计量与实际视频文件全部通过检查。完整数据的副本缺少视频时正确终止读取。uint8 与浮点 RGB observation 均在实际记录上逐帧验证，来源 SHA256 保持一致，编码线程全部关闭。

`student_recorded_inference` 通过共享 `load_policy` 读取已有 `v3_student_a8cc2f8` 权重，在实际 Lift 记录上完成 243 帧 CPU 推理。报告保存模型与 episode SHA256、控制周期、动作范围，以及 student 的 `grasp_v3` 和来源记录的 `grasp_v4`。此项验证范围包含记录观测上的模型推理、双相机输入与动作转换。

采集和回放状态位于 `teleop/sim_state.py`，频率检查位于 `teleop/timing.py`，metadata 位于 `il/datasets/validation.py`。场景录制使用共享 Isaac 启动和 `ExitStack` 关闭处理；原生程序检查与保存初始状态的策略验证包含应用关闭处理。模型加载要求 observation/action processors 完整。

MuJoCo 完成四个 episode、1,000 个控制步骤和 20,000 个物理步骤，速度与力矩限制通过，任务成功为 0/4。最终 PPO 的既有独立评估为 0/100。此次 GPU 启动数量为零，本项目 GPU 进程查询为空。`native_physics_verified`、`rl_policy_success_verified`、`hardware_run_verified` 与 `full_v2_verified` 保持各报告规定的独立状态；原生运行、成功策略 sim2sim、视觉 student 独立任务和真实视频恢复继续等待对应验收。模型、训练日志和原始记录全部保留。
