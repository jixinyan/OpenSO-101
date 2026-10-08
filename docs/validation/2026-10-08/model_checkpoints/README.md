# 完整模型 checkpoint 与 CPU 流程验证

源码版本为 `c7c95e0e2f527e7215741b59a31bda0394d87b80`。全部 18 项 CPU 阶段通过，独立 Linux CI 通过 123 项检查；实际 sdist、wheel 和 192 个模块、835 处项目 import 检查通过。报告及其 SHA256 见 [verification.json](verification.json)。

## 保存和读取

ACT 的完整参数数量为 51,597,190，Diffusion 为 266,622,758。两个模型使用实际 328 帧双相机数据执行 CPU forward、backward、inference 和 normalization 检查。准备目录保留完整 `pretrained_model/`，包含权重、配置与两个 processors 的状态文件。

保存配置使用计划设备 `cuda:0`，实际模型参数来自 CPU 初始化，optimizer 更新次数为零。共享加载入口指定 CPU，并检查完整权重。每个模型的 180 个 processor 状态 Tensor 逐项相同；模型状态 SHA256 保持一致，动作推理最大误差为零。完整记录见 [ACT](act.json) 和 [Diffusion](diffusion.json)，两个文件的 SHA256 与 [汇总报告](suite/il_training/report.json) 一致。

## 输入检查

八项无效配置、十六项 checkpoint 输入通过实际程序终止检查。文件检查包括缺少权重、两个 processor 配置及 normalization 状态文件；路径与设备检查包括不存在的本地 `Path`、文件路径、无效 Hub repo_id 和不可用 CUDA。`il play` 与 `il eval` 使用实际 checkpoint 的不完整副本，全部在启动 Isaac 前终止。实际模型与数据文件 SHA256 保持一致。

## 其他阶段

双相机直接 LeRobot 采集、HDF5 checkpoint、遥操控制、键盘 IK、场景、视频标定、已有 student 和动作监督完成 CPU 运行。MuJoCo 策略完成四个 episode、1,000 个控制步骤、20,000 个物理步骤，实际速度与力矩检查通过，任务成功为 0/4。原生 Isaac、人工成功采集和独立策略成功继续等待相应验收，GPU 计算与 RL 训练保持停止。
