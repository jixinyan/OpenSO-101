# v2 训练与视觉策略

## 安装与运行环境

使用 Python 3.11、Isaac Lab 2.3.0、Isaac Sim 5.1.0。依赖包含 rsl_rl 3.0.1、SB3 / sb3-contrib 2.7.0、skrl 1.4.3 和 rl_games 1.6.5。

```bash
uv pip install -e '.[scenes]' --torch-backend cu128 \
  --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match
```

CUDA capability `sm_103` 的主机使用 `--override requirements-sm103.txt` 安装 NVRTC 12.9.86。`jd_B300` 已运行验证；机器报告的 GPU 型号为 NVIDIA H20G。

## 统一训练配置

将以下内容保存为 `train.json`：

```json
{
  "backend": "rsl_rl",
  "algo": "ppo",
  "seed": 42,
  "iterations": 2000,
  "rollout_steps": 96,
  "epochs": 5,
  "mini_batches": 4,
  "hidden_dims": [256, 128, 64],
  "learning_rate": 0.0001
}
```

```bash
openso101 rl train --task OpenSO101-CustomScene-v0 --scene outputs/apple_usd \
  --backend rsl_rl --train-config train.json --num_envs 64 \
  --output outputs/apple_ppo --logger tensorboard --headless --no-video
openso101 rl eval --task OpenSO101-CustomScene-v0 \
  --checkpoint outputs/apple_ppo --n-episodes 100 --num-envs 16 --headless
```

`--backend` 支持 `rsl_rl`、`sb3`、`skrl` 和 `rl_games`。PPO 可使用全部 backend；SAC 与 TQC 使用 SB3。统一入口使用 TensorBoard。`--video` 保存训练视频，`--with-cameras` 启用双相机观测，`--visual-dr` 使用任务配置提供的视觉随机化。

训练目录保存实际环境配置、训练配置、模型、归一化状态、适用的 replay buffer、源码压缩包及 SHA256 清单。自定义场景保存完整副本。没有 Git metadata 的远程代码副本通过 `--source-revision` 或 `OPENSO101_SOURCE_REVISION` 提供完整源代码 commit SHA；实际执行的 Python 文件同时保存在 `source.zip`。

```bash
openso101 rl train --task OpenSO101-CustomScene-v0 --backend sb3 \
  --train-config train.json --resume --load_run outputs/previous_run \
  --output outputs/continued_run --logger tensorboard --headless --no-video
```

继续训练读取完整训练目录，检查任务、算法、backend、文件内容及场景版本。输出目录必须为新目录。评估直接读取 `checkpoint.json` 选择正确的加载方式。

## 动作与相机随机化

训练环境启用动作延迟、deadband 和执行幅度变化，各环境分别维护并重置历史。双相机训练启用相机安装位置和旋转随机化。play 与遥操采用对应任务的评估配置。

PickPlace 包含抓取、抬升、搬运与释放后的稳定放置。阶段推进要求抓取接触，完成条件要求夹爪释放、速度满足限制并保持稳定。

## 视觉蒸馏与部署入口

```bash
openso101 rl distill --teacher-run outputs/apple_ppo --output outputs/apple_student \
  --num-envs 16 --iterations 1500 --rollout-steps 16 --headless
```

teacher 使用 rsl_rl PPO 的状态观测。student 输入 wrist / overhead RGB 与六个关节的本体观测，图像统一处理为 64 × 64。导出包含 `student.pt`、`student.json`、模型 hash、关节动作转换和控制周期。现有 `sim2real deploy --policy-path` 可以读取该目录；部署频率需要与 `student.json` 的 `control_dt` 一致。

## 已执行检查

2026-09-26 在 `jd_B300` 执行：

- 四个 backend 的 PPO 短程训练、保存、加载与独立评估。
- SB3 的 SAC、TQC 短程训练、保存、加载与独立评估。
- 双相机视觉蒸馏，导出模型后读取实际录制的 12 帧图像执行推理。
- PickPlace 的四个并行环境、100 次 reset 与 200 个控制步骤。

短程训练用于验证程序流程，评估成功率为 0。收敛训练、多随机种子成功率、真机部署与任务性能尚未验收。Agent 编排结构与实际模型服务连接独立于本文中的训练接口。
