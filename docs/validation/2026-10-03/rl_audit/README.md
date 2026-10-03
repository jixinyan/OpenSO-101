# RL 与场景服务运行检查

## 来源

原生物理检查来自固定提交 `2ef2ebd`，物理周期为 0.002 秒，控制周期为 0.02 秒。每种条件包含四个真实 Isaac 环境、500 个控制步骤；速度直接记录自 PhysX 每个物理步骤。

| 任务与条件 | 物理采样最高速度 rad/s | 目标转换最大误差 rad | 速度检查 |
|---|---:|---:|---|
| Lift nominal | 1.782182 | 0 | 通过 |
| PickPlace nominal | 2.101363 | 0 | 未通过 |
| Lift randomized | 1.500001 | 0 | 通过 |
| PickPlace randomized | 1.553141 | 0 | 通过 |

完整 episode 的折扣 shaping 累计绝对值均低于 `7e-7`。检查报告保留实际 trace 和源码 SHA256。以上记录只证明报告中的采样范围；任务成功未通过。

## 训练、导出与 MuJoCo

训练来源为 `85af781`：Lift、32 个环境、两次 PPO 更新，共 6,144 transitions。独立评估使用 seed `10042`，100 episodes 的成功率为零，接近率为 19%，抓取与持续提升率为零。完整逐 episode 数据保存在 `evaluation_history.json`。

checkpoint SHA256 为 `fa287ff2614db8f93e9867ab9b491a799070a46175ec5120582a3adb4249f365`。导出的策略在四个原生 Isaac 环境执行 300 步，观测重建误差为零，actor 和目标转换最大误差均为 `1.1920928955e-7`。

同一策略使用官方 `so101_old_calib.xml`、实际源环境物理参数、CoACD 夹爪碰撞和 OSQP 受限 implicit PD，在 MuJoCo 3.14.0 执行四个 episode，共 10,000 个物理步骤。实际速度与力矩检查通过，成功率为零。1,200 个末端坐标的最大位置误差为 `1.9275e-6` m，最大旋转误差为 `1.0678e-5` rad。策略与 source trace SHA256、实体参数和每个 episode 的检查结果保存在 `mujoco_report.json`。

## 模型服务

`gpt-6-astra` 的真实 Responses 请求经过完整 SSE 完成事件和 Pydantic schema 检查，保留输入任务与释放要求。请求用量为 153 input tokens、46 output tokens，共 199 tokens。请求和响应 SHA256 保存于 `model_completion.json`，记录不包含密钥。预算按实际 HTTP 请求次数计算。

CPU 检查为 44 项通过，四项使用替代服务或对象的测试未执行。
