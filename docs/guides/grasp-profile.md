# 抓取训练配置

`grasp_v2` 用于 Lift 和 PickPlace，通过 `--task-profile grasp_v2` 选择。`checkpoint.json` 保存 `task_profile`；继续训练、独立评估、策略导出与视觉蒸馏读取该字段。未提供字段的已有 checkpoint 使用 `default`。继续训练时改变 `task_profile` 会终止。

夹爪目标为 `clamp(0.4 + 0.4 * action, 0, 0.8)`，单位为 rad。手臂目标保持为默认关节位置加上 `0.5 * action`。共享动作转换包含处理后的限幅，MuJoCo 与视觉 student 使用相同定义。

抓取位置在 `gripper` 局部坐标中计算，中心为 `(0.01, 0, -0.09)`，与实际 `ee_frame` 相同；三个方向的尺度分别为 4cm、2.5cm、3cm。该几何奖励描述接近抓取中心的程度；实际持物状态仍通过两侧夹爪各大于 0.5N 的接触力判定。

| reward | 权重 | 条件与数值 |
|---|---:|---|
| approach | 1 | 尚未抓取时的 `1-tanh(distance/0.2)` |
| alignment | 1 | 三个方向距离的 Gaussian score |
| closure | 2 | 抓取位置 score 乘以闭合指令比例；进入释放范围后停用 |
| closure_away | -0.5 | 尚未抓取时，远离抓取中心的闭合指令 |
| grasp_hold | 8 | 两侧夹爪实际接触 |
| held_height | 12 | 抓取时物体高于桌面 1.5cm 的高度进度，达到 11.5cm 时为 1 |
| held_goal | 16 | 抓取时的 `1-tanh(goal_distance/0.10)` |
| processed_action_change | -0.1 | 处理后六关节目标的变化平方和；reset 步骤归零 |
| joint_vel | -0.0001 | 关节速度平方和 |
| success_bonus | 100 | 实际 success termination |
| release | 12 | PickPlace 最终阶段进入 3cm 放置范围时的打开指令比例 |
| stable_placement | 20 | PickPlace 释放后的稳定保持时间除以 0.5 秒，最大为 1 |

RewardManager 将各项乘以控制周期 0.02 秒。权重为实验参数，任务收敛需要实际训练和独立评估验证。

Lift 完成需要物体高度超过 4cm、进入距离目标 5cm 的范围，并连续保持双夹爪接触 0.25 秒。PickPlace 完成仍需要完整抓取、搬运和放置阶段，打开夹爪后在 3cm 目标范围持续稳定 0.5 秒。MuJoCo 读取导出的任务配置并执行对应条件。

训练配置使用 `gamma=0.995`，在 50Hz 控制下有效折扣时间约四秒。物理和观测随机化继续启用，物体位置范围保持不变。运行配置、全部源码和环境配置随模型保存。

```bash
openso101 rl train --task OpenSO101-Lift-v0 --algo ppo --backend rsl_rl \
  --task-profile grasp_v2 --train-config configs/rl/grasp_v2.json \
  --num_envs 1024 --output outputs/lift_grasp_v2 --logger tensorboard --headless --no-video
```
