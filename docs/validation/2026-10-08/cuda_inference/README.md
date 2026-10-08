# CPU 功能与 CUDA 推理入口验证

源码版本为 `b84a3bb0e1b6bad2f21761dd1ca10941995ff9a3`。统一流程的全部 19 项 CPU 阶段通过，独立 Linux CI 通过 123 项检查和四项 CUDA 请求检查。192 个模块、844 处项目 import、全部未定义名称、实际 sdist 与 wheel 检查通过。全部 30 份正式报告及其 SHA256 保存在 [verification.json](verification.json)。

## 采集与模型

同步与异步 LeRobot 直接录制分别验证全部 336 帧、三个 episode，包含 checkpoint 恢复、取消、重新打开和追加记录。动作与状态误差为零，已有数据与视频 SHA256 保持一致，工作线程全部关闭。HDF5 的磁盘与内存 checkpoint 完成全部 29 个状态字段恢复。遥操每步目标限制、恢复姿态保持、键盘 IK 和双相机导出使用实际保存数据完成检查。

ACT、Diffusion 完整模型的参数数量分别为 51,597,190 和 266,622,758，CPU forward、backward、inference 与 optimizer、scheduler 构建通过。完整权重与 processors 保存在准备目录。每个模型重新加载后的 180 个 processor 状态 Tensor 完全相同，模型状态 SHA256 一致，动作推理误差为零。optimizer 更新次数为零，IL 训练未启动。[ACT](models/act.json)、[Diffusion](models/diffusion.json) 和 [汇总报告](suite/il_training/report.json) 保存实际配置、权重文件 SHA256 和来源。

八项无效配置和十六项 checkpoint 输入完成实际终止检查。`il play` 与 `il eval` 的不完整模型在启动 Isaac 前终止。已有视觉 student 完成实际 243 帧双相机记录的 CPU 推理，模型 profile 与记录 profile 分别保存。

## CUDA 入口

Linux 的 CUDA 视觉策略执行使用单张空闲物理 GPU 2、实际设备 UUID 和进程监督。共享模型加载要求有效启动记录、正确进程来源和单个可见 CUDA 设备。Torch 的逻辑设备使用 `cuda:0`；CPU 推理可以独立运行。

实际禁止 CUDA 的 CPU 进程拒绝四项 GPU 请求，检查前后均未初始化 CUDA。统一流程与独立 CI 分别保存 [CPU 入口报告](suite/cuda_inference_entry.json) 和 [CI 入口报告](ci_cuda_inference_entry.json)。实际 GPU 推理、设备竞争和真机设备访问继续等待对应验收。

## 场景与 MuJoCo

十个场景 bundle、实际 CPU renderer 视频标定、既有策略历史与动作监督通过 CPU 检查。已有 MuJoCo 策略完成四个 episode、1,000 个控制步骤、20,000 个物理步骤；速度、力矩与 FK 检查通过，任务成功为 0/4。[GPU 进程报告](suite/gpu_inventory.json) 确认本项目没有 GPU 作业，其它项目保持运行。

GPU 计算与 RL 训练保持停止。完整 v2 的任务成功、原生 Isaac 场景恢复、人工键盘成功采集、RL 收敛和成功策略 sim2sim 继续等待对应运行。报告保持 `gpu_tests_started=false`、`native_physics_verified=false` 和 `full_v2_verified=false`。
