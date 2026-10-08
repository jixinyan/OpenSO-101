# 采集回放与 CPU 验证

源码版本为 `7b84a59167e80507153877230d15d85324c73de8`。统一流程通过全部 20 个 CPU 阶段，独立 Linux CI 通过 129 项检查、四项 CUDA 请求检查，以及实际 sdist 与 wheel 验证。源码检查包含 198 个模块和 887 处项目 import。全部 31 份正式报告及其 SHA256 保存在 [verification.json](verification.json)。

## 录制状态

真实 PickPlace episode 的全部 328 帧、23 个状态字段通过 CPU 转换检查。使用记录的四个环境原点完成 1,312 次物体状态转换，最大误差为 `5.960464477539063e-8` 米；quaternion、线速度和角速度逐项一致。来源 episode SHA256 保持不变。实际回放入口在启动 Isaac 前拒绝不一致的任务名称。[状态报告](suite/recorded_state.json) 保存来源、程序 SHA256 和测量结果。

Stack 的 HDF5 录制保存 `cube_top_root_state`、`cube_bottom_root_state`、`cube_top_was_lifted` 和 `task_episode_step`。物体恢复读取记录的环境原点；回放报告保存完整帧数、观察到的任务成功和验收状态。共享代码位于 `teleop/state_records.py`、`teleop/sim_state.py` 与 `teleop/replay_validation.py`。Stack 的实际原生录制与恢复继续等待 GPU 验收。

同步与异步 LeRobot 直接录制分别通过 336 帧、三个 episode 检查，包含 checkpoint、取消、重新打开与追加记录。既有 HDF5 的磁盘与内存 checkpoint 完成全部 29 个保存字段恢复。来源数据与视频保持原值，工作线程全部关闭。

## 模型与场景

ACT、Diffusion 的完整 CPU 模型分别包含 51,597,190 和 266,622,758 个参数。forward、backward、保存、processors 恢复和实际动作推理全部通过。每个模型使用 48 帧真实双相机与关节记录，在三个环境、两个批次执行 16 次连续动作推理，共享入口与实际 LeRobot 推理的最大误差为零。模型保存 50 Hz、256×256 双相机、六个 SO-101 关节名称、动作单位和 metadata SHA256；optimizer 更新次数为零，IL 训练未启动。

统计检查读取实际四环境轨迹，验证精确 episode 配额、批次完成、结束步骤和中断状态。来源结束步骤分别为 328、329、349、286。新的策略任务成功继续等待独立评估。[模型与统计报告](suite/il_evaluation/report.json)、[ACT](models/act.json)、[Diffusion](models/diffusion.json) 保留完整来源。

十个已有场景、CPU renderer 视频标定、键盘 IK、已有视觉 student 推理和策略历史检查通过。MuJoCo 已保存策略完成四个 episode、1,000 个控制步骤、20,000 个物理步骤；实际速度与力矩限制通过，任务成功为 0/4。[GPU 进程报告](suite/gpu_inventory.json) 确认本项目没有 GPU 作业，其它项目保持运行。

## 原生验收

统一 GPU 阶段包含九个步骤：原生程序检查、Lift 与 PickPlace checkpoint、ACT 与 Diffusion 推理、Stack 与 CustomScene checkpoint、十个场景运行，以及保存策略的实际初始状态检查。四类场景的 checkpoint 检查包含保存文件的全部八帧恢复；CustomScene 保存对应来源 bundle。

GPU 计算与 RL 训练保持停止，原生步骤尚未执行。来源物理参数匹配、人工键盘任务成功、RL 收敛、成功策略 sim2sim 与真实视频恢复继续等待对应验收。报告保持 `gpu_tests_started=false`、`native_physics_verified=false` 和 `full_v2_verified=false`。
