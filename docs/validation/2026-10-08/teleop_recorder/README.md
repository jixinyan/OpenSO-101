# 遥操控制与直接 LeRobot 采集验证

源码版本为 `efbe232ae25881d09c7c490143e8631be925a177`。统一流程通过全部 18 项 CPU 阶段；GitHub CI 通过 123 项检查和实际 sdist、wheel 验证。源码检查覆盖 191 个模块、831 处项目 import。全部报告的 SHA256 保存在 [verification.json](verification.json)。

## 采集和控制

同步与异步 LeRobot 直接录制分别读取全部 336 帧、三个 episode。数据来自已有 328 帧双相机记录；重新打开数据集后追加八帧。动作与状态的 motor units 误差为零，最大时间误差为 `1.1444091807533141e-7` 秒。wrist、overhead 视频的归一化像素平均绝对误差分别为 `0.008717655204236507` 和 `0.005949870217591524`。

checkpoint 恢复删除后续相机帧，取消录制清除未保存 episode，重新打开数据集保持已有 Parquet 与视频 SHA256，保存时保留输入相机数组的内容。同步与异步工作线程全部关闭。完整记录见 [LeRobot 采集报告](suite/lerobot_recorder/report.json)。

HDF5 的磁盘与内存 checkpoint 完成全部 29 个状态字段恢复，数据与实际来源一致。遥操目标的每步变化限制、恢复姿态保持、NumPy/Torch 转换以及无效输入终止通过实际保存动作检查。报告分别见 [HDF5 checkpoint](suite/recorder_checkpoint/report.json) 和 [控制检查](suite/teleop_controls.json)。

## 模型和场景

ACT、Diffusion 完整模型通过 CPU forward、backward、inference 以及实际 optimizer、scheduler 构建；参数数量分别为 51,597,190 和 266,622,758。optimizer 更新次数为零，IL 训练未启动。[ACT](act.json) 和 [Diffusion](diffusion.json) 保存配置、模型状态 SHA256 与数据来源；报告 SHA256 与 [汇总记录](suite/il_training/report.json) 一致。

其他阶段使用已有场景、视频标定、双相机数据和模型执行 CPU 验证。MuJoCo 策略完成四个 episode、1,000 个控制步骤、20,000 个物理步骤，实际速度与力矩检查通过，任务成功为 0/4。

## 原生验收

Lift、PickPlace、Stack 的完整 Isaac 场景 checkpoint 验证代码已纳入 GPU 阶段。统一报告保持 `gpu_tests_started=false`、`native_physics_verified=false` 和 `full_v2_verified=false`。GPU 计算与 RL 训练保持停止；人工成功采集与独立策略成功需要对应运行。
