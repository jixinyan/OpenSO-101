# grasp_v3 训练与验收

## 控制和观测

`configs/rl/grasp_v3.json` 使用 RSL PPO、Tanh Gaussian、观测归一化和 adaptive learning rate。actor 的采样、确定性输出、log probability 和 entropy 共同采用 Tanh 变换。策略导出、MuJoCo 和视觉 student 使用相同动作转换。

每个控制步骤以当前测得的六个关节位置为参考，目标增量为 `clip(action, -1, 1) × 0.04 rad`。目标受关节范围与夹爪 `[0, 0.8] rad` 范围限制，全部物理步骤使用同一个位置目标。实际关节超出目标范围时，目标限制会增加修正量。实际速度使用每个物理步骤的测量值进行验收。

控制周期为 0.02 秒，物理周期为 0.002 秒。机器人 actuator 使用官方 SO-101 MJCF 中的 armature `0.028`、stiffness `17.8`、damping `0.6` 和 effort limit `3.35`。这些是仿真参数，真机参数需要校准。solver velocity iterations 为 8，maximum depenetration velocity 为 0.1 m/s，`solve_articulation_contact_last=True`。

policy 观测包含关节位置和速度、物体位置和速度、任务目标、双侧抓取状态、上一个实际动作、任务阶段、保持时间和剩余 episode 时间。Lift 为 38 个数值，PickPlace 为 34 个数值。达到 episode 时间上限作为有限时间任务的终止，critic 不执行 timeout bootstrap。

## Reward 和成功条件

进展奖励为 `5 × (gamma × Phi(next) - Phi(previous))`，其中 `gamma` 与训练配置一致。`Phi` 包含接近、抓取中心、闭合、双侧接触、提升、目标距离和 PickPlace 阶段。episode 起始与终止的势能均为零；完整 episode 的折扣 shaping 累计需要通过实际轨迹检查。

每次成功奖励为 100。`Phi` 的上限为 18，终止步骤的 shaping 大小最多为 90。关节速度和目标变化分别按控制时间计算惩罚。成功事件同时要求当前步骤的 success termination 和当前 termination 标记。

Lift 要求双侧接触力均超过 0.5 N、物体高度超过 4 cm、目标距离小于 5 cm，并保持 0.25 秒。PickPlace 要求完成抓取和搬运阶段，释放后位于目标 3 cm 范围，夹爪位置超过 0.4 rad，物体线速度不超过 0.02 m/s、角速度不超过 0.1 rad/s，并稳定保持 0.5 秒。PickPlace 的保持时间在当前物理步骤计算。

## 独立评估和模型保存

每 100 iterations 保存 checkpoint，并启动独立 Isaac 进程完成 100 episodes 确定性评估。评估使用训练 seed 加 10000，记录每条 episode 的成功、长度、return 和任务阶段。全部模型、训练日志、source archive、环境配置和评估报告保留。

`model_best.pt` 按独立成功率选择。`model.pt` 保存最后完成的 iteration。每个 seed 的连续三次评估均达到 90% 时，该 seed 完成训练验收。

`rl campaign` 管理 Lift 和 PickPlace 各三个 seed，在六个独立 GPU 上运行；`campaign.json` 保存进程、配置、来源与独立验收状态。各任务的三个 seed 都满足条件后，多个 seed 验收才能成立。选定模型保存为独立 `selected_policy` snapshot。

```bash
openso101 rl campaign --train-config configs/rl/grasp_v3.json \
  --output outputs/rl_progress/grasp_v3_campaign \
  --seeds 42 43 44 --gpus 0 1 2 3 4 5 --num-envs 2048
```

训练、导出、MuJoCo 任务成功和视觉 student 任务成功分别使用各自的真实运行报告。程序启动、有限数值和动作转换检查各自记录其范围。

## 当前检查

2026-10-03 已获得新的训练授权，全部已有模型与评估记录保留。44 项现有 CPU 检查通过。原生 Lift 的四环境、300 控制步骤检查包含 3000 个物理步骤，最高关节速度为 1.167577 rad/s；动作转换误差为零，四条完整 episode 的折扣 shaping 累计绝对值低于 `1.6e-7`。其余环境条件、训练收敛和成功策略 sim2sim 正在进行实际验收。
