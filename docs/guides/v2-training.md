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
  --backend rsl_rl --algo ppo --train-config train.json --num_envs 64 \
  --output outputs/apple_ppo --logger tensorboard --headless --no-video
openso101 rl eval --task OpenSO101-CustomScene-v0 \
  --checkpoint outputs/apple_ppo --n-episodes 100 --num-envs 16 --headless
```

`--backend` 支持 `rsl_rl`、`sb3`、`skrl` 和 `rl_games`。PPO 可使用全部 backend；SAC 与 TQC 使用 SB3。统一入口使用 TensorBoard。`--video` 保存训练视频，`--with-cameras` 启用双相机观测，`--visual-dr` 使用任务配置提供的视觉随机化。

训练目录保存实际环境配置、训练配置、模型、归一化状态、适用的 replay buffer、源码压缩包及 SHA256 清单。自定义场景保存完整副本。没有 Git metadata 的远程代码副本通过 `--source-revision` 或 `OPENSO101_SOURCE_REVISION` 提供完整源代码 commit SHA；实际执行的 Python 文件同时保存在 `source.zip`。

```bash
openso101 rl train --task OpenSO101-CustomScene-v0 --backend sb3 --algo ppo \
  --train-config train.json --resume --load_run outputs/previous_run \
  --output outputs/continued_run --logger tensorboard --headless --no-video
```

继续训练读取完整训练目录，检查任务、算法、backend、文件内容及场景版本。输出目录必须为新目录。评估直接读取 `checkpoint.json` 选择正确的加载方式。

rsl_rl PPO 使用 log 参数表示探索标准差，每个 iteration 保存中间模型。`CheckedPPO` 检查观测、动作、分布、梯度和参数的有限性，并保存实际更新记录。继续训练沿用原模型的 policy 配置，保持网络结构和观测归一化设置。

评估按照环境分配 episode 数量，完整完成请求的数量后生成报告。Lift 和 PickPlace 报告包含接近物体、两侧夹爪接触、物体高度、持物抬升及 PickPlace 阶段统计；这些诊断在控制步骤开始前采样。任务成功率读取实际 success termination，报告同时保存模型 SHA256、训练和评估代码版本、训练 transitions、运行设备及 Torch 版本。

## 动作与相机随机化

训练环境启用动作延迟、deadband 和执行幅度变化，各环境分别维护并重置历史。双相机训练启用相机安装位置和旋转随机化。play 与遥操采用对应任务的评估配置。

PickPlace 包含抓取、抬升、搬运与释放后的稳定放置。阶段推进要求抓取接触，完成条件要求夹爪释放、速度满足限制并保持稳定。

## 视觉蒸馏与部署入口

```bash
openso101 rl distill --teacher-run outputs/apple_ppo --output outputs/apple_student \
  --num-envs 16 --iterations 1500 --rollout-steps 16 --headless
```

teacher 使用 rsl_rl PPO 的状态观测。student 输入 wrist / overhead RGB、六个关节的本体观测与任务目标的 robot root frame 米制 xyz，图像统一处理为 64 × 64。导出包含 `student.pt`、`student.json`、模型 SHA256、关节动作转换、控制周期和经过文件校验的 teacher 副本。每个蒸馏 iteration 保存中间模型。

```bash
openso101 rl student-eval --student outputs/apple_student \
  --num-envs 16 --n-episodes 100 --seed 10042 --headless \
  --recording-output outputs/apple_student_recording
openso101 sim2real validate --policy-path outputs/apple_student \
  --episode outputs/apple_student_recording/episodes/episode_000000.hdf5 \
  --output outputs/apple_student_inference
```

独立评估使用 teacher 对应的任务配置和实际相机，保存每个 episode 的实际 termination 与模型来源。采集保存首个环境的一条完整 episode，包含双相机、实际关节目标、物体状态和当前任务目标。离线验证读取全部帧，并通过部署所用的动作转换生成 motor commands。

```bash
openso101 rl validate-loop --teacher-run outputs/teacher_snapshot \
  --output outputs/policy_validation_loop \
  --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
  --collision-bundle outputs/rl_progress/gripper_collision \
  --mujoco-python /path/to/mujoco/environment/bin/python
```

闭环入口保存经过 SHA256 校验的 teacher 副本，依次执行 teacher 独立评估、实际 Isaac 策略导出、MuJoCo 任务评估、双相机 student 蒸馏、student 独立评估和完整采集文件推理。三个任务评估各包含 100 episodes，均要求至少成功 90 次。导出使用另一个独立 seed 的 100 个实际初始环境；MuJoCo 使用源环境的实际物理参数、CoACD gripper 和速度、力矩受限的 implicit PD。

每个步骤保存命令、日志、退出状态、实际成功次数和文件 SHA256。任务成功次数不足时终止后续步骤并保留当前记录。`single_policy_loop_verified` 只记录单个策略的完整闭环任务结果；三个 seed 的训练验收与真机验收仍各自记录。

`sim2real deploy --policy-path` 可以读取 student 目录；部署频率需要与 `student.json` 的 `control_dt` 一致。带有目标输入的 student 通过 `--goal-file` 读取当前目标 JSON。目标文件为包含三个有限数值的 JSON 数组，使用 robot root frame 米制 xyz。真机验收根据用户要求暂缓。

## 已执行检查

2026-09-26 在 `jd_B300` 执行：

- 四个 backend 的 PPO 短程训练、保存、加载与独立评估。
- SB3 的 SAC、TQC 短程训练、保存、加载与独立评估。
- 双相机视觉蒸馏，导出模型后读取实际录制的 12 帧图像执行推理。
- PickPlace 的四个并行环境、100 次 reset 与 200 个控制步骤。

2026-10-03 的视觉 student 使用实际 `8965174` teacher，完成两次蒸馏 iteration、128 个真实 transitions；独立任务评估为 `0/8`。原生评估采集的 250 帧双相机 HDF5 通过完整文件检查和全部帧的模型推理、动作转换检查，控制频率为 50 Hz。报告位于 [recorded_inference.json](../validation/2026-10-03/rl_audit/student/recorded_inference.json)。

短程训练验证程序流程。收敛训练、多随机种子成功率、成功 student 和真机部署仍需要各自的任务成功记录。
