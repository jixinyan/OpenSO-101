# LeRobot 直接采集与模型来源验证

源码版本为 `707ee0b158d9aefd3e513d8adf81f8d1bdf1e489`。统一流程通过全部 20 个 CPU 阶段，独立 Linux CI 通过 129 项检查、四项 CUDA 请求检查及实际 sdist、wheel 验证。源码包含 200 个模块和 930 处项目 import。全部 31 份报告及 SHA256 保存在 [verification.json](verification.json)。

## 直接采集

检查读取已有 PickPlace 的 328 帧双相机 HDF5，分别保存两个 164 帧片段，并在重新打开数据集后追加前八帧。同步与异步模式各完成 336 帧、三个 episode 的实际保存与全部帧读取，动作和关节转换误差为零。checkpoint 恢复、相机文件裁剪、取消和 writer 关闭均通过；已有数据与视频 SHA256 保持一致。[采集报告](suite/lerobot_recorder/report.json) 保存全部来源与测量结果。

`il record --record-format lerobot` 的 `meta/openso101_recording.json` 保存来源物理设置和每个 episode 的帧数、任务与成功标记。实际来源为 50 Hz、`physics_dt=0.001`、`decimation=20`、`grasp_v4`、`nominal` 和 `reward_discount=0.99`。不一致的 FPS、仿真来源与无效成功标记均通过实际终止检查；取消和追加保留已有来源记录。

## 模型来源与连续推理

ACT、Diffusion 完整模型直接读取上述实际 LeRobot 数据。CPU forward、backward、保存、processors 恢复与推理全部通过，IL optimizer 更新次数为零。模型保存 256×256 双相机、六个 SO-101 关节名称、动作单位、实际物理设置，以及 `dataset_recording_sha256=94ca093c6bc435886465a57a9307f2ce6f42ab2a3a9a02e157fea58bec6110e8`。[ACT](models/act.json) 与 [Diffusion](models/diffusion.json) 保存完整检查结果。

每个模型使用 48 帧实际记录，在三个环境、两个批次中完成 16 次连续动作推理。共享入口与 LeRobot 的最大动作误差为零，模型与 processors 文件保持不变。三个无效任务或输出请求在启动 Isaac 前终止。[推理报告](suite/il_evaluation/report.json) 保存精确 episode 统计、来源和动作比较。模型采用 CPU 初始化参数，独立任务成功继续等待训练与实际评估。

```mermaid
flowchart LR
    Source[实际双相机与关节记录] --> Recorder[直接 LeRobot 采集]
    Recorder --> Metadata[物理设置与 episode metadata]
    Metadata --> Model[ACT 与 Diffusion 完整 CPU 检查]
    Model --> Settings[保存模型的来源 SHA256]
    Settings --> Request[任务与场景请求检查]
    Request --> Native[等待原生 GPU 验收]
```

## 原生检查与任务验收

CustomScene 的直接采集保存对应场景副本、SHA256 和相对路径。Lift、PickPlace、Stack 和 CustomScene 的原生检查入口包含 HDF5 与直接 LeRobot 来源比较，资源通过 `ExitStack` 关闭。对应原生运行尚未执行。

已有场景、视频标定、键盘 IK、记录状态转换和视觉 student 的 CPU 检查通过。MuJoCo 已保存策略完成四个 episode、1,000 个控制步骤和 20,000 个物理步骤，速度与力矩限制通过，任务成功为 0/4。[策略报告](suite/mujoco_policy/report.json) 保存来源及结果。

[GPU 进程报告](suite/gpu_inventory.json) 确认本项目没有 GPU 作业，其它项目保持运行。GPU 计算与 RL 训练保持停止。九项原生步骤、成功策略 sim2sim、人工键盘任务成功、真实视频恢复和真机验收继续等待对应运行；报告保留 `gpu_tests_started=false`、`native_physics_verified=false` 与 `full_v2_verified=false`。
