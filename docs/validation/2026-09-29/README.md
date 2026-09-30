# Isaac RL 与终端键盘运行记录

## 当前执行状态

定时任务 `openso-101` 已通过 Codex 应用删除。按用户最新指令，本轮完成实际物理参数记录与实体比较后停止；RL 训练保持停止，真机工作暂缓。

按用户指令，Lift 与 PickPlace 的 RL 训练均已停止，启动或恢复 RL 训练需要用户重新授权。两项训练目录中的 `model_0.pt`、`model_50.pt`、`model_100.pt` 已重新读取，全部模型参数有限，文件 SHA256 保存于 [停止状态检查](training_stopped.json)。已有记录覆盖键盘采集、LeRobot 导出与回放、MuJoCo sim2sim、sim2real 和 agentic real2sim。

sim2sim 的已有记录包括 `01f075e` 使用实际 Isaac 关节目标、初始位置和速度，在 MuJoCo 对 Lift 四环境各 250 步、PickPlace 四环境各 400 步进行动力学比较。两项最大关节位置差异为 0.71237 和 0.23731 rad，出现在 0.08 和 0.10 秒；实际速度及参数来源见 [动力学比较说明](../../guides/sim2sim.md#相同动作的动力学比较)。此比较保持任务成功与物理等价未验证。

- [Lift 原始比较报告](lift_dynamics_comparison_report.json)
- [PickPlace 原始比较报告](pick_place_dynamics_comparison_report.json)
- [六项实际输入拒绝检查](comparison_guards_report.json)

两项保存模型的新导出各完成四环境、500 步实际 Isaac 推理，记录每步实际物理参数和物体速度。MuJoCo 的名义 PD 与实际 PD 检查使用相同源轨迹和初始状态，全部关节目标与 Isaac 状态逐项完全一致。实际 PD 下最大关节位置差异为 Lift 0.63094 rad、PickPlace 0.28084 rad；最大速度为 13.37341 和 7.03054 rad/s。当前保持物理等价与任务成功未验证。

- [Lift 推理与动作转换](lift_physics_export_report.json)
- [PickPlace 推理与动作转换](pick_place_physics_export_report.json)
- [Lift 名义 PD](lift_physics_nominal_report.json)
- [Lift 实际 PD](lift_recorded_pd_report.json)
- [PickPlace 名义 PD](pick_place_physics_nominal_report.json)
- [PickPlace 实际 PD](pick_place_recorded_pd_report.json)
- [实际输入配对检查](pd_comparison_pairs_report.json)

## 原生速度限制的配对运行

`51647ee` 在同一实际 Isaac 场景中对 Lift 与 PickPlace 各运行四组配对环境、100 个控制步骤。原生接口检查每组质量、惯性、材质、关节参数和每步 PD 一致，初始关节位置与速度误差为 0。每组使用相同已记录关节目标，原生 solver 速度限制分别为 2 和 1000 rad/s。

2 rad/s 组的最大关节速度约为 2.014 rad/s；1000 rad/s 组的 Lift 与 PickPlace 分别为 19.46912 和 9.69941 rad/s，配对关节位置最大差异为 0.54092 和 0.26580 rad。2 rad/s 组与源轨迹的最大关节位置差异分别为 0.000253 和 0.000257 rad。该运行确认速度限制对当前轨迹的影响；源场景随机物理参数完整恢复、跨模拟器物理等价和任务成功均保持未验证。

运行入口为 `scripts/check_isaac_velocity_limit.py`，执行代码 SHA256 为 `f2f28d4ba76ae380593fc8edb0317daaa8c888950c36d9e8293ccefca2b9900c`。主机与本地均保留 `outputs/rl_progress/lift_velocity_limit/` 和 `pick_place_velocity_limit/` 中的实际 HDF5 与原始报告。主机执行脚本为 `outputs/rl_progress/run_velocity_experiment.sh`，日志为 `velocity_experiment.log`。物理周期 0.01 秒，控制周期 0.02 秒；使用直接物理步骤。

六项真实无效输入请求均在启动 Isaac 前终止：单步骤、无效速度限制、非有限速度限制、源轨迹数量不足、episode 边界和已有输出目录。

- [Lift 原生实验报告](lift_native_velocity_report.json)
- [PickPlace 原生实验报告](pick_place_native_velocity_report.json)
- [输入拒绝检查](native_velocity_guards_report.json)
- [运行方式与后续目标](../../guides/sim2sim.md#原生-isaac-速度限制实验)

## MuJoCo velocity servo 运行

`4485a89` 使用原生 velocity servo 与受限速度目标，保留 effort limits 和物理求解。Lift 与 PickPlace 各完成四环境，分别为每环境 250／400 个控制步骤、2500／4000 个物理步骤。相同源轨迹的 position PD 控制组重新运行完成，实际输入逐项一致。

Lift 的最大关节位置误差由 0.63094 减少至 0.16383 rad，PickPlace 由 0.28084 减少至 0.13260 rad。每个物理步骤记录的最大速度分别为 2.01763 和 2.21527 rad/s。速度目标公式与记录的误差为 0，原生 actuator 力矩反馈公式误差小于 `4.5e-16` N·m，全部速度目标与力矩范围检查通过。实际速度超过目标限制的行为保留在报告中，solver 约束等价、物理等价与任务成功仍未验证。

完整 HDF5 与报告保存在本地和主机 `outputs/rl_progress/lift_velocity_servo_final/`、`pick_place_velocity_servo_final/`、`lift_position_pd_control/`、`pick_place_position_pd_control/`。检查脚本为同目录下的 `check_velocity_servo_results.py` 和 `check_velocity_servo_guards.py`，报告保留其代码 SHA256。四项无效输入请求使用实际文件检查，覆盖已有输出、单步骤、超过源记录数量和旧轨迹缺少速度限制字段。

- [Lift 原始报告](lift_velocity_servo_report.json)
- [PickPlace 原始报告](pick_place_velocity_servo_report.json)
- [Lift position PD 控制组](lift_position_pd_control_report.json)
- [PickPlace position PD 控制组](pick_place_position_pd_control_report.json)
- [配对输入与原生力矩检查](velocity_servo_pairs_report.json)
- [四项输入拒绝检查](velocity_servo_guards_report.json)
- [运行方式](../../guides/sim2sim.md#mujoco-受限速度目标驱动)

## 实际实体物理参数记录与比较

`2c8a990` 在 `jd_B300` 使用两项保存模型，各完成四环境、500 个控制步骤，观测重建误差为 0，策略与动作目标误差分别小于 `8.4e-7` 和 `3e-7`。每步记录机器人与物体的原生质量、惯性、COM、材质、实体姿态和场景重力。代码、实体顺序与坐标定义保存在导出报告和 HDF5 metadata 中。

本地 MuJoCo 3.14.0 的 `sim2sim physics` 对每个任务比较 2000 个关节状态、14,000 个实体状态。COM 最大位置差异为 2.72／2.62 μm，按质量归一化后的惯性相对差异低于 `9.3e-6`；完整惯性差异最高为 0.18805，实际质量与名义 MJCF 质量之比为 0.81195–1.18196。报告使用共同 robot-root 坐标系比较完整惯性，保留局部坐标方向差异。实际 armature、关节摩擦、物体质量、重力与材质均已记录；它们对运动的单独影响和接触等价需要后续实验。

原始输出保存在本地和主机的 `outputs/rl_progress/lift_body_physics_verified_50/`、`pick_place_body_physics_verified_50/`，比较报告为 `lift_body_physics_verified_report.json` 和 `pick_place_body_physics_verified_report.json`。执行脚本为 `run_body_physics_export.sh`，原生日志为主机的 `body_physics_verified_export.log`；检查脚本为 `check_body_physics_results.py` 与 `check_body_physics_guards.py`。已有输出、缺少物理字段的旧记录通过真实 CLI 请求拒绝检查。

- [Lift 实际导出报告](lift_body_physics_export_report.json)
- [PickPlace 实际导出报告](pick_place_body_physics_export_report.json)
- [Lift 实体比较](lift_body_physics_report.json)
- [PickPlace 实体比较](pick_place_body_physics_report.json)
- [数量、有限值与来源检查](body_physics_records_report.json)
- [两项实际输入拒绝检查](body_physics_guards_report.json)
- [入口与当前后续目标](../../guides/sim2sim.md#实体物理参数比较)

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
- [180 帧 PickPlace 状态采集与恢复](keyboard_state_report.json)
- [正式入口 PickPlace 回放报告](replay_cli_state_report.json)
- [正式入口自定义场景回放报告](replay_cli_custom_report.json)
- [正式入口旧键盘数据回放报告](replay_cli_legacy_report.json)
- [正式入口参数检查报告](replay_cli_guards_report.json)
- [键盘操作指南](../../guides/teleop.md)

## 重复运行

`eb3e763` 的正式 `il replay --report` 完成三项实际 Isaac 检查，共 132 帧；三项进程的 exit code 均为 0。PickPlace 状态数据从第 120 帧恢复并执行 60 帧，六项场景字段及关节位置、速度恢复误差均为 0。自定义场景执行全部 12 帧，实体状态和关节恢复误差均为 0。旧键盘数据执行前 60 帧，检查其已保存的关节状态。三项控制周期与源 FPS 一致，动作误差为 0，每帧实际关节和双相机数据均通过检查，源任务成功标记均为 `false`。报告包含检查代码 SHA256，任务成功和完整物理状态重现保持未验证。

实际运行脚本保存在主机的 `outputs/rl_progress/run_native_replay_report.sh`。已有报告文件、空动作范围与负数保持步骤使用同一实际 HDF5 检查，均在启动 Isaac 前以非零状态终止；检查脚本为 `check_replay_cli_guards.py`。正式命令示例：

```bash
/home/jixin/workspace/envs/openso101-v2/bin/python -m openso101.cli.main il replay \
  --episode outputs/rl_progress/keyboard_state_dataset/episodes/episode_000000.hdf5 \
  --checkpoint-frame 120 --start-frame 120 --stop-frame 180 --hold-steps 0 \
  --report outputs/replay_state_report.json --headless --no-camera-viewports
```

PickPlace 的状态采集检查使用实际双相机、机器人、物体和 command，录制 180 帧后重置环境，再恢复第 120 帧并回放后续 60 帧。源动作来自已有键盘 HDF5，图像、关节和场景状态在本次真实运行中重新采集。报告中的六项场景状态与两项关节状态恢复误差均为 0，放置计时器为 0，任务成功为 `false`。物理随机化参数及接触求解状态的重现仍需独立验证。

实际运行脚本为 `outputs/rl_progress/run_state_recording.sh` 与 `verify_state_recording.py`，HDF5 保存在 `outputs/rl_progress/keyboard_state_dataset/episodes/episode_000000.hdf5`，代码版本为 `539b22e`。采集器要求所有帧的状态字段与首帧一致；记录到的放置计时器随 HDF5 状态恢复。

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

场景编译与双相机运行入口版本为 `69c1ee6`。已有 Apple bundle 和真实 Astra 生成的 `so101_apple_move_right` 均完成四个环境、100 次 reset 和 200 个控制步骤；每个相机检查 800 帧，观测形状为 `(4, 34)`，episode 终止数量为 0。场景准备报告保存源 manifest、编译清单、runtime 报告和检查代码 SHA256，并核查请求数量与实际数量一致。

生成调用使用已有仿真录制视频的两个采样帧、本地资产目录和离线资产检索配置，产生一项生成苹果资产，模型状态为 `needs_review`。完整模型结果保存在主机 `outputs/rl_progress/agent_runtime_prepared/agent_loop_result.json`；场景 SHA256 为 `f78efad8d55471727dd61b8db9dfc98ce68d8984aa9c305cfe3e68dc1ce1c537`。`runtime_verified` 保留程序检查范围，模型审查、任务接触、可达性、物体可见性和成功采集使用对应验收。

生成场景的双相机另行录制各 720 帧、512×512、60 FPS。机器人使用已有键盘 HDF5 的前 720 帧绝对关节目标，每步 ActionManager 动作与源动作的误差阈值为 `1e-6`，相机和状态检查通过。录制报告保存场景、源键盘 HDF5、相机视频和录制代码的 SHA256；任务成功与真机运行保持未验证。

主机运行脚本为 `run_scene_prepare.sh`、`run_agent_runtime.sh`、`run_agent_capture.sh` 和 `capture_agent_scene.py`，均保存在 `outputs/rl_progress/`。正式报告：

- [已有 Apple 场景准备](apple_preparation_report.json)
- [已有 Apple 场景运行](apple_prepare_runtime_report.json)
- [模型生成场景准备](agent_preparation_report.json)
- [模型生成场景运行](agent_prepare_runtime_report.json)
- [模型生成场景双相机录制](agent_scene_capture_report.json)
- [完整流程 MP4 检查](agent_demo_report.json)

## Agentic 场景生成演示

`outputs/rl_progress/agent_demo/OpenSO101-agentic-demo.mp4` 为 35 秒、1920×1080、30 FPS 的完整演示，全部 1050 帧经过实际解码检查。内容依次展示用户任务与视频输入、资产查询与生成、任务场景布局、Isaac 双相机运行和验收状态。场景和相机视频使用上述真实运行产物；输入为已有仿真视频，机器人动作来自已有键盘 HDF5，任务成功与真机运行保持未验证。

演示 SHA256 为 `ee5cc580d0c7fbc5c69105e221875676de09d65657c0fddefa1f03b5851c04f9`。模型结果、录制报告与构建脚本 SHA256 保存在 [演示检查报告](agent_demo_report.json)。MP4 同时保存于 `jd_B300` 的 `outputs/rl_progress/OpenSO101-agentic-demo.mp4`。

录制脚本为仓库中的 `scripts/capture_agent_scene.py`，在配置好的 Isaac 环境中使用编译场景、真实动作记录与新的输出目录执行：

```bash
PYTHONPATH=src /home/jixin/workspace/envs/openso101-v2/bin/python \
  scripts/capture_agent_scene.py \
  outputs/rl_progress/agent_runtime_prepared/compiled \
  outputs/rl_progress/keyboard_dataset/episodes/episode_000001.hdf5 \
  outputs/new_scene_capture --steps 720
```

`scripts/build_agent_demo.py` 使用 PyAV、Pillow 和 macOS 的 STHeiti 字体生成演示。输入目录需要包含实际运行的 `bundle/`、`prepared/`、`input_frames/` 和 `capture/`；脚本检查场景及视频 SHA256，要求两个输入采样帧和双相机各 720 帧，输出文件存在时立即终止。使用包含这些文件的新目录执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/build_agent_demo.py outputs/my_demo
```

## 相机与部署检查

sim2real 相机与停止控制的代码版本为 `9d1d711`。正式 `camera-check` 使用已有自定义场景的双相机视频，各读取全部 12 帧，metadata 为 128×128、60 FPS，图像转换使用部署入口相同的函数。视频使用 FFmpeg 从已保存的 AV1 编码转换为 H.264；全部解码帧数量、形状和 FPS 一致，归一化 RGB 平均绝对误差分别约为 0.00518、0.00192。两种编码的源文件和输出文件 SHA256 均保留。上述相机运行与 student 录制数据推理的进程均正常退出。

真实保存的 student 用于停止文件、30／60Hz 控制频率和零步骤配置检查，通过实际函数调用观察确认均未连接 follower。停止文件存在时没有加载模型；控制频率检查实际加载 student 并拒绝不匹配的频率。共享初始姿态使用实际 Torch 和动作转换函数，生成时没有导入 Isaac。检查范围为程序与录制文件，真机运行保持未验证。

运行脚本与报告保存在主机 `outputs/rl_progress/`：`run_camera_student_check.sh`、`transcode_camera_files.sh`、`check_camera_transcode.py` 和 `check_deploy_preflight.py`。正式报告：

- [双相机读取](camera_read_report.json)
- [视频编码转换](camera_transcode_report.json)
- [部署预检查](deploy_preflight_report.json)

共享策略的数值比较、动作诊断、MuJoCo 配对场景和视觉 student 检查见 [接口运行说明](../../guides/sim2sim.md)，reward 判断见 [RL audit](../../guides/rl-audit-2026-09-29.md)。实际模型和完整轨迹保留在主机及本地的 `outputs/rl_progress/`，本目录保存其配置与原始 JSON 报告。

Python 3.11 下 44 项检查通过，覆盖实际 PTY、文件、MP4、HDF5、OpenUSD、Torch 运算和 CLI 参数。四项含替代服务或对象的测试未执行。

```bash
OPENSO101_SKIP_ISAAC=1 TMPDIR="$PWD/outputs/tmp" PYTHONPATH=src \
  .venv/bin/python -m pytest tests/scenes tests/test_cpu_regressions.py tests/test_cli_rl.py \
  -k 'not eagerly and not model_service and not agent_loop_materializes' \
  --basetemp=outputs/pytest-cpu-hold -q
```
