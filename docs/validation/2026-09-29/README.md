# Isaac RL 与终端键盘运行记录

## 当前执行状态

按用户指令，Lift 与 PickPlace 的 RL 训练均已停止，持续开发任务禁止启动或恢复 RL 训练。两项训练目录中的 `model_0.pt`、`model_50.pt`、`model_100.pt` 已重新读取，全部模型参数有限，文件 SHA256 保存于 [停止状态检查](training_stopped.json)。当前继续推进键盘采集、LeRobot 导出与回放、MuJoCo sim2sim、sim2real 和 agentic real2sim。

## 运行环境与模型

- 主机：`jd_B300`，NVIDIA H20G；Isaac Sim 5.1、Isaac Lab 2.3、Torch 2.7.0+cu128。
- 仓库：`/home/jixin/workspace/code/OpenSO-101-v2`。
- Python：`/home/jixin/workspace/envs/openso101-v2/bin/python`。
- Lift 与 PickPlace：RSL PPO、seed 42、1024 个训练环境、200 次迭代、每次 rollout 96 步，每项共 19,660,800 transitions。
- MLP：256/128/64；观测归一化；learning rate 0.0001；entropy coefficient 0.005；每次更新 5 epochs、4 minibatches。
- 探索标准差使用 `noise_std_type=log`；每 50 次迭代保存 checkpoint，保留最终 `model.pt`。
- 训练代码版本：`3a16ff38f5ecba4ab2dcc0f6d58fd5ab39f7c3f1`。原始评估报告记录写入报告时的仓库版本 `dda1cc2`。

各运行目录中的 `train.json`、`backend.json`、`checkpoint.json` 为主机原始文件。`learning_curves.json` 由 TensorBoard 的 EventAccumulator 读取实际事件文件生成，每项 scalar 含 200 个有限值；`step_unit` 标明迭代编号或运行秒数。模型、环境配置、source.zip 和原始事件文件保留在主机 `outputs/rl_progress/` 对应目录。三个最终模型 SHA256 已在主机重新计算，与本目录评估报告及 checkpoint manifest 一致。

## 独立评估

每项任务重新启动 Isaac 并加载最终模型，使用 seed 42、64 个环境、100 个 episode。环境 0–35 各分配两个 episode，环境 36–63 各分配一个 episode。报告数量、环境分配、有限 reward、成功率及全部阶段统计均已读取并核查。

| 指标 | Lift | PickPlace |
|---|---:|---:|
| 成功 episode | 0/100 | 0/100 |
| `reached` | 90% | 100% |
| `grasped` | 9% | 0% |
| `lifted` | 1% | 2% |
| `held_above_table` | 0% | 0% |
| `carry_stage` | — | 0% |
| `place_stage` | — | 0% |

`reached` 表示末端与物体中心距离小于 0.08m；`grasped` 使用实际双夹爪接触判定；`lifted` 表示物体高于环境原点 0.04m；`held_above_table` 要求接触判定与高度条件同时满足。阶段标记在控制步骤之前采样，并在各 episode 内累计。任务成功使用环境的 success termination。

物体高度变化尚未形成持物抬升，当前策略未达到收敛验收要求。收敛验收要求三个 seed、每个 seed 连续三次 100-episode 评估成功率至少 90%；RL 训练当前保持停止。

原始报告：

- [Lift](evaluation-20260929T233508184521Z.json)
- [PickPlace](evaluation-20260929T233148413539Z.json)

## grasp_v2 保存模型的独立检查

Lift 与 PickPlace 均完成四环境、300 步实际动作检查，连续夹爪目标覆盖 0–0.8rad，转换后的关节目标与实际 ActionManager 误差为 0，关节状态与 reward 有限。训练使用动作随机化；比较读取随机化后的实际输入。第 50 次迭代 snapshot 各包含 5,013,504 transitions，训练源码为 `cdcaf0e`，独立评估与策略导出使用 `bc59081`。文件复制过程中核查 checkpoint SHA256，并保留 source.zip、配置和运行 metadata。

| 独立评估指标 | Lift | PickPlace |
|---|---:|---:|
| 成功 episode | 0/100 | 0/100 |
| reached | 53% | 100% |
| grasped | 0% | 0% |
| lifted | 0% | 0% |
| held_above_table | 0% | 0% |

两项 TorchScript actor 完成四环境、500 步实际比较，观测重建误差为 0，策略动作误差小于 `8.4e-7`。MuJoCo 对相同已保存模型各执行四个完整 episode，成功率均为 0/4；两项 2000 个姿态的最大位置误差分别约 2.07μm、1.99μm。

- [Lift 实际动作检查](grasp_v2_runtime/lift.json)
- [PickPlace 实际动作检查](grasp_v2_runtime/pick_place.json)
- [Lift snapshot metadata](grasp_v2_snapshot_50/lift.json)
- [PickPlace snapshot metadata](grasp_v2_snapshot_50/pick_place.json)
- [Lift 独立评估](evaluation-20260930T010530915791Z.json)
- [PickPlace 独立评估](evaluation-20260930T010445856266Z.json)
- [策略导出与 MuJoCo 运行范围](../../guides/sim2sim.md)

## 继续训练检查

旧版 scalar-std RSL 自定义场景模型完成两次新增迭代，共新增 64 transitions，总计 128。保存的新模型经独立进程重新加载并完成两个 episode 的评估。旧 policy 配置保持一致；配置与 hash 保存在 `compat_resume/`，评估 seed 为 0，成功率为 0/2。

[继续训练评估报告](evaluation-20260929T232605247734Z.json)

## 终端键盘与采集

SSH 实际 TTY 输入经过 prompt_toolkit、KeyboardDevice、实际 Jacobian IK 和 Isaac 关节控制器，接收 `UP`、`PAGE_DOWN`、`A`、`G`、`SPACE`、`Q`。六个关节均产生有限的实际运动，全部动作目标满足关节限位。

双相机均为 128×128 RGB，每帧检查有限值及图像变化。录制的 2281 帧 HDF5 已通过 episode 校验，文件保留在主机 `outputs/rl_progress/keyboard_dataset/episodes/episode_000001.hdf5`。其 `task_success=false`。该文件验证控制与记录流程，成功任务示范仍需独立验收。

正式 `il export` 将此 episode 的全部 2281 帧导出到 `outputs/rl_progress/keyboard_lerobot_verified`。使用 `--include-failures --skip-leading-frames 0`，源数据的成功标记保持 `false`。实际 LeRobotDataset 逐帧解码双相机，并与源 HDF5 比较六关节动作和观测，全部帧的动作与观测误差均为 0。视频压缩后的归一化像素平均绝对误差约为 wrist 0.00852、overhead 0.00595。`meta/openso101_export.json` 保存源 SHA256、帧数、成功标记和筛选配置。

读取环境使用主机已有的 `outputs/ffmpeg/root/usr/lib/x86_64-linux-gnu`、其 `pulseaudio`、`blas`、`lapack` 子目录与 graphics 目录。实际运行脚本保留在 `outputs/rl_progress/read_keyboard_export.sh`。

正式 `il replay` 使用同一源 HDF5 完成 2281 帧录制动作，加上 30 个初始保持步骤，共 2311 个实际 Isaac 控制步骤，控制周期为 1/60 秒。源关节位置与速度恢复误差为 0，每步动作与源记录一致；2311 次关节状态检查与每台相机的 2311 次数据检查均通过，进程正常退出，SSH exit code 为 0。`outputs/rl_progress/verify_keyboard_replay.py` 使用 `sys.settrace` 在实际函数返回时读取状态，保存报告；它调用正式 CLI 并保持实际环境和控制器。

该源文件没有记录物体与任务目标状态，报告中的 `source_object_state_available` 与 `source_command_goal_available` 均为 `false`。此次验证范围为关节动作和双相机运行，完整场景重现与任务成功仍需包含相应状态的录制数据。

正式 `il record --keyboard-input terminal --headless --no-record` 入口接收方向、夹爪开合与退出输入，日志包含 `Quit-and-discard requested`，进程正常退出，SSH exit code 为 0。窗口按键与真机 leader 操作仍需实际操作验收。

在 `2f52a02` 代码版本下，另一次无相机 Isaac 运行完成 2382 次无按键检查，关节目标最大变化为 0rad；环境重置后调用 `reset_reference()`，控制参考误差为 0rad。

正式 `il record` 的独立双相机运行通过实际终端 `UP`、`C`、`G`、`R` 输入。第 386 帧保存 checkpoint，夹爪操作后恢复到第 386 帧；恢复完成后立即读取，关节位置、关节速度、物体状态、任务目标、放置计时器，以及键盘 arm/jaw 控制参考误差均为 0。此次放置计时器值为 0。随后 `Q` 丢弃当前 episode 并正常退出，SSH exit code 为 0，任务成功保持未验证。

- [双相机控制与记录报告](keyboard_report.json)
- [关节目标保持与重置报告](keyboard_hold_report.json)
- [C/R 原始状态观察](keyboard_cli_restore_report.json)
- [C/R 数值验证](keyboard_cli_restore_verified.json)
- [2281 帧 LeRobot 读取报告](keyboard_lerobot_report.json)
- [2281 帧 Isaac 关节与双相机回放](keyboard_replay_report.json)
- [键盘操作指南](../../guides/teleop.md)

## 重复运行

在上述主机仓库目录使用已配置环境执行。`TMPDIR` 指向 `outputs/tmp`，Isaac 使用已下载的 SO-101 USD；EULA 已得到用户授权接受。CUDA NVRTC 与图形库目录在 `LD_LIBRARY_PATH` 中设置为对应环境路径。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

/home/jixin/workspace/envs/openso101-v2/bin/python -m openso101.cli.main rl train \
  --task OpenSO101-Lift-v0 --algo ppo --backend rsl_rl \
  --train-config docs/validation/2026-09-29/lift_seed42/train.json \
  --num_envs 1024 --output outputs/rl_repeat/lift_seed42 \
  --logger tensorboard --headless --no-video

/home/jixin/workspace/envs/openso101-v2/bin/python -m openso101.cli.main rl eval \
  --task OpenSO101-Lift-v0 --checkpoint outputs/rl_repeat/lift_seed42 \
  --n-episodes 100 --num-envs 64 --seed 42 --headless
```

PickPlace 使用 `OpenSO101-PickPlace-v0`、`pick_place_seed42/train.json` 和独立输出目录。

## CPU 检查

共享策略的数值比较、动作诊断、MuJoCo 配对场景和视觉 student 检查见 [接口运行说明](../../guides/sim2sim.md)，reward 判断见 [RL audit](../../guides/rl-audit-2026-09-29.md)。实际模型和完整轨迹保留在主机及本地的 `outputs/rl_progress/`，本目录保存其配置与原始 JSON 报告。

Python 3.11 下 44 项检查通过，覆盖实际 PTY、文件、MP4、HDF5、OpenUSD、Torch 运算和 CLI 参数。四项含替代服务或对象的测试未执行。

```bash
OPENSO101_SKIP_ISAAC=1 TMPDIR="$PWD/outputs/tmp" PYTHONPATH=src \
  .venv/bin/python -m pytest tests/scenes tests/test_cpu_regressions.py tests/test_cli_rl.py \
  -k 'not eagerly and not model_service and not agent_loop_materializes' \
  --basetemp=outputs/pytest-cpu-hold -q
```
