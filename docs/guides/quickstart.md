# 快速操作

安装与环境要求见 [安装指南](install.md)。当前 GPU 计算与 RL 训练保持停止，实际验收记录见 [v2 状态](v2-status-2026-10-08.md)。在仓库根目录运行命令；每次验证使用尚未存在的输出目录，保存全部报告与来源文件。

## CPU 验证

`jd_B300` 的 `native` 与 `mujoco` 环境支持以下操作。源码检查验证 Python import、Shell 语法和命令帮助。

```bash
OPENSO101_REPO="$PWD" bash scripts/run_cpu_python.sh mujoco \
  -m openso101.cli.main validate source \
  --output outputs/rl_progress/source_check
```

统一 CPU 流程读取 `configs/validation/v2_preparation.json` 中指定的实际模型、数据与资产，完成场景、视频标定、键盘 IK、采集恢复、LeRobot、IL 模型和 MuJoCo 检查。

```bash
OPENSO101_REPO="$PWD" bash scripts/run_cpu_python.sh mujoco \
  -m openso101.cli.main validate run configs/validation/v2_preparation.json \
  --phase cpu --output outputs/rl_progress/v2_cpu_validation
```

本地 LeRobot 数据可以独立检查完整 ACT 或 Diffusion 模型：

```bash
OPENSO101_REPO="$PWD" bash scripts/run_cpu_python.sh native \
  -m openso101.cli.main il train --policy act \
  --dataset outputs/my_lerobot_dataset --prepare-only \
  --preparation-output outputs/rl_progress/il_preparation/act_check
```

## GPU 验收与训练

GPU 启动与进程检查见 [GPU 使用](gpu-usage.md)。重新授权后，统一 GPU 验收使用通过 CPU 检查的相同源码、配置、数据与输出目录；每个程序最多使用一张空闲的物理 GPU 2。原生验收包含场景、Lift、PickPlace、Stack checkpoint 和保存策略的独立运行。

训练入口见 [训练与视觉策略](v2-training.md) 和 [IL 配置与模型](il-training.md)。成功率读取实际任务 termination，训练曲线、模型文件和独立评估分别保存。当前已有 PPO 的独立评估结果为 0/100；MuJoCo 已保存策略的验证结果为 0/4。

## 键盘与采集

图形界面的键盘采集方式如下，需要已授权的 GPU 运行环境：

```bash
openso101 il record --task OpenSO101-PickPlace-v0 \
  --teleop-device keyboard --keyboard-input window \
  --repo-root outputs/rl_progress/keyboard_pick_place
```

`S` 保存成功记录并退出，`Q` 取消记录并退出，`C` 保存 checkpoint，`R` 恢复 checkpoint 与姿态保持。SSH 终端的控制方式、按键和双相机设置见 [遥操作指南](teleop.md)。

HDF5 采集保存双相机、动作、本体观测与仿真状态。LeRobot 导出、数据读取和策略模型使用统一的关节动作转换。CPU 程序验证、人工键盘成功采集和原生状态恢复分别保存验收结果。

## 功能入口

[代码目录](code-map.md) 与 [脚本目录](../../scripts/README.md) 列出各功能的位置。场景入口见 [场景管理](scene-management.md) 和 [自定义场景](custom-scenes.md)；策略迁移见 [sim2sim](sim2sim.md)。
