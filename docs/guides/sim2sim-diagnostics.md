# 保存策略的 sim2sim 轨迹诊断

`sim2sim diagnose` 在 CPU 上读取实际 Isaac 导出与 MuJoCo 评估，检查动作转换、初始状态、夹爪接触、关节控制误差和末端距离。程序保存 JSON 与 `diagnostics.png`。全部输入文件需要通过原始报告中的 SHA256 检查。

```bash
OPENSO101_REPO="$PWD" bash scripts/run_cpu_python.sh mujoco \
  -m openso101.cli.main sim2sim diagnose \
  --policy outputs/rl_progress/v4_seed44_99_b49353f_portable \
  --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
  --evaluation outputs/rl_progress/scene_preflight_20261011/suite/mujoco_policy \
  --output outputs/rl_progress/sim2sim_diagnostics
```

输出目录需要尚未存在，并且与策略、评估和机器人资产目录独立。已有模型、轨迹和图表保持原有内容。

## 检查范围

| 项目 | 检查方式 |
|---|---|
| 来源 | 策略、metadata、官方 XML、Isaac 与 MuJoCo 轨迹分别检查 SHA256 |
| 关节与末端 | 官方 MJCF 在 CPU 上执行 forward kinematics，与全部原生姿态比较 |
| 末端参考点 | 与实际 Isaac `ee_object_distance` 逐帧比较 |
| 起始状态 | 两种模拟器的关节位置和物体位置需要一致 |
| 动作含义 | 使用同一 `PortablePolicy.joint_targets` 检查保存的 raw action 与关节目标 |
| 接触 | 每个步骤的两个夹爪接触力均超过来源阈值，才计入双侧接触 |
| 时间 | Isaac 仅使用第一个 episode 的记录，终止与超时标记确定其范围 |
| 图表 | 分别绘制两个模拟器的夹爪目标、实际关节位置、双侧接触力和末端距离 |

Isaac 原始导出中的 GPU actor 与 CPU portable model 使用 `1e-5` 数值容差；MuJoCo 同一 CPU 动作转换使用 `1e-6` 容差。JSON 同时保存实际误差与对应容差。

两种模拟器从相同初始环境分别执行策略，之后的控制轨迹允许不同。图表按各自的实际控制时间绘制。动力学等价和完整任务成功需要对应的独立验收。

## 检查入口

`scripts/check_sim2sim_diagnostics.py` 使用真实轨迹运行完整诊断，并在副本上检查来源 SHA256、动作转换、字段帧数、初始状态、已有输出保护和目录关系。统一流程使用 `configs/validation/v2_preparation.json` 中的 `mujoco_diagnostics` CPU 阶段。

功能实现位于 `src/openso101/sim2sim/diagnostics.py`，命令参数位于 `src/openso101/cli/sim2sim.py`。当前控制误差使用动作发出之前的关节状态与该步骤的关节目标计算。
