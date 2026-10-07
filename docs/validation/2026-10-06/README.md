# 2026-10-06 至 2026-10-07 实际验证记录

## 双相机成功数据

| 文件 | 实际检查 | 结果 |
|---|---|---|
| [lift_capture.json](lift_capture.json) | Isaac scripted controller、四个实际环境 | Lift 4/4 |
| [lerobot_validation.json](lerobot_validation.json) | 完整成功 episode 的双相机 LeRobot 读取 | 243/243 帧、动作与关节观测转换误差为零 |
| [lift_replay.json](lift_replay.json) | 四环境录制到单环境的原生任务回放 | 243/243 帧、恢复与动作误差为零、实际 Lift 成功 |

原始数据位于 `jd_B300` 的 `outputs/rl_progress/lift_portable_dataset_20261006`。源 HDF5 SHA256 为 `34e06fb6e73f0b384357f8799eaa294e38e78911a0c9ef28c49b1644ca267fb0`；回放 HDF5 SHA256 为 `846873377ce913cbab470a9aaf3fd37371136d79ed044650420d34e7d174089f`。控制周期为 0.02 秒，物理周期为 0.001 秒。回放的 `task_success_verified=true`，完整物理轨迹重复性尚待验收。

## 保存模型的真实评估

| 文件 | 范围 | 结果 |
|---|---|---|
| [demonstration_initialization.json](demonstration_initialization.json) | 840 个实际成功示范样本、8000 gradient steps | bounded action MSE 4.64e-5 |
| [initial_policy_evaluation.json](initial_policy_evaluation.json) | 初始化模型独立 100 episodes | 成功 0/100、接近物体 33/100 |
| [evaluation_history.json](evaluation_history.json) | 八次 PPO 更新后独立 100 episodes | 成功 0/100、接近物体 34/100 |
| [initial_policy_paired.json](initial_policy_paired.json) | 来源成功 episode 的完全相同初始状态 | 250 步、任务未成功 |
| [initial_policy_paired.initial_observation.json](initial_policy_paired.initial_observation.json) | 实际恢复后的 38 维观测 | 最大误差为零 |
| [perturbed_expert.json](perturbed_expert.json) | 0.015 rad 控制扰动下的真实 expert 轨迹 | Lift 3/4 |

示范拟合与任务成功各自验收。扰动轨迹使用 `expert_policy_action` 作为 actor 标签，`policy_action` 保存实际执行动作，reward 使用真实执行 transition。监督准备只选择完整成功 episode。

![实际示范动作拟合](figures/demonstration_fit.png)

![保存模型的独立任务评估](figures/independent_evaluation.png)

## 规划与程序验证

[pick_place_plan.json](pick_place_plan.json) 记录四个实际环境的 PickPlace waypoint 与完整路径采样结果，四个环境均通过运动学检查。该报告中的 `continuous_collision_path_verified=false` 和 `task_success_verified=false` 保留各自的验收范围。

[planner_geometry.json](planner_geometry.json) 比较 MuJoCo 的 840 个真实 recorded poses 和 2759 个 contacts，最大 pose 与 contact 误差均为零，缓存重新构建的模型参数完全一致。

[native_regressions.json](native_regressions.json) 保存 Isaac 原生环境的实际 pytest 结果与源码版本，进程内部的检查结果为零。对应原生日志包含 44 项通过与四项未执行。

全部 GPU 工作限定物理 GPU 2。当前源码、继续训练和待完成验收见 [开发状态](../../guides/v2-status-2026-10-06.md)。
