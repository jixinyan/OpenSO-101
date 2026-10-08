# IL 模型仿真设置与分批评估检查

源码版本为 `bb145ef2a049c5a96fd209765b52ba7b5fdbb84f`。统一流程通过全部 20 个 CPU 阶段，独立 Linux CI 通过 129 项检查、四项 CUDA 请求检查、实际 sdist 与 wheel 验证。源码检查包含 197 个模块和 878 处项目 import。全部 31 份正式报告与 SHA256 保存在 [verification.json](verification.json)。

ACT 与 Diffusion 的完整模型通过 CPU forward、backward、保存、重新加载和连续动作推理。每个模型读取实际 episode 的 48 帧双相机与关节数据，使用三个环境、两个批次执行 16 次 `select_action`。共享推理入口与实际 LeRobot processors 的动作结果逐项一致，最大误差为零。权重和来源数据保持原值，optimizer 更新次数为零，IL 训练未启动。[评估报告](suite/il_evaluation/report.json) 保存实际参数、文件 SHA256 和来源。

模型目录保存 `openso101_simulation.json`，包含真实数据的 50 Hz、256×256 双相机、六个 SO-101 关节名称、`motor_units` 和 metadata SHA256。完整 checkpoint 的设置写入和读取通过实际检查。仿真入口在创建环境之前检查频率和图像尺寸。

评估按照精确 episode 数量和每个环境的配额运行；当前批次结束后重置策略历史。结束记录在物理步骤之后、环境 reset 之前保存。既有四环境轨迹的结束步骤分别为 328、329、349、286；统计读取这些实际记录，检查一个、三个、四个、五个和七个 episode 的分配与中断状态。实际 Isaac 执行、独立策略任务成功和来源物理参数匹配继续等待对应验收。

双相机 LeRobot 数据读取、同步与异步导出、直接录制和 checkpoint 检查全部通过。MuJoCo 已保存策略完成四个 episode、1,000 个控制步骤、20,000 个物理步骤，速度与力矩限制通过，任务成功为 0/4。十个已有场景与 CPU 视频标定继续保存独立记录。

GPU 计算和 RL 训练保持停止。本项目 GPU 进程查询为空。正式报告保持 `gpu_tests_started=false`、`native_physics_verified=false` 和 `full_v2_verified=false`。
