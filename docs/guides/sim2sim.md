# 共享策略、MuJoCo 与 sim2real 检查

## 导出并验证策略

`rl export` 使用 Isaac Lab exporter 保存包含观测归一化的 TorchScript actor，从实际环境读取观测顺序、默认关节状态、动作 scale/offset、binary 开合参数、关节限位、控制周期、nominal PD 和任务参数。

```bash
openso101 rl export --task OpenSO101-Lift-v0 \
  --checkpoint outputs/rl_progress/lift_seed42 \
  --output outputs/lift_portable --num-envs 4 --validation-steps 500 --headless
```

每个验证步骤使用实际关节状态、物体 robot-root 坐标、任务目标、接触力及上一步原始动作重建观测，比较原始策略与导出模型，并比较动作转换与实际 ActionManager。误差超过 `1e-5` 时立即终止。输出包括 `policy.pt`、`policy.json`、`validation.json` 和 `isaac_validation.hdf5`。

HDF5 中的状态和当前动作在控制步骤之前记录，reward 和终止标记对应随后发生的环境 transition。文件记录末端姿态、双侧接触力、关节状态、开合指令和 reward，可用于 RL 行为诊断。

## MuJoCo 评估

安装 `openso-101[sim2sim]`，使用 MuJoCo 3.14.0。机器人模型来自 [TheRobotStudio 的 SO-101 仿真目录](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/README.md)，本次使用版本 `5f6d2b876a53a4872e405b991dd925556c9e38a4` 的 `so101_old_calib.xml` 与其 meshes。该目录提供两种零点定义；此入口使用经过实际坐标验证的 old-calibration 定义。

```bash
openso101 sim2sim mujoco --policy outputs/lift_portable \
  --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
  --episodes 4 --output outputs/mujoco_lift
```

入口由 MuJoCo MjSpec 加载官方 MJCF，添加桌面和 3cm 物体，使用导出的 nominal PD 和 effort limits。`Pitch` 的 MuJoCo 坐标为 Isaac 坐标减去 π/2，`Elbow` 加上 π/2，其余关节坐标相同。policy 的控制周期保持为 0.02 秒，MuJoCo 物理周期为 0.002 秒。

运行前对 Isaac HDF5 中的全部关节状态检查末端位置与 quaternion；位置误差超过 1mm 或旋转误差超过 0.001rad 时终止。评估使用记录第一帧的实际关节位置、速度、物体姿态和任务目标，每个 Isaac 环境对应一个配对初始场景。`episodes` 数量必须不超过该记录的环境数量。

输出 `trajectory.hdf5` 与 `report.json`，包含实际任务结果、关节坐标误差、模型与 mesh hashes、策略 hash、初始数据 hash、控制周期及轨迹 hash。Lift 使用导出的高度和目标距离条件；PickPlace 检查实际双夹爪接触、抬升与搬运阶段，以及释放后的稳定放置。

两种模拟器的接触求解、碰撞网格、速度限制、惯性和摩擦行为仍需独立比较。MuJoCo 保留官方 MJCF 的惯性与 frictionloss，使用其 convex meshes；Isaac 的 2rad/s 关节速度限制及物理随机化尚未在 MuJoCo 复现。`physics_equivalence_verified=false` 保留在报告中。

本次 MuJoCo 在本地 Mac CPU 运行，Isaac 轨迹来自 `jd_B300`。Lift 和 PickPlace 各运行四个完整 episode，均为 0/4。每项坐标检查覆盖 2000 个状态；最大位置误差分别约 2.37μm、2.03μm，最大旋转误差约 `1.2e-5`rad。该结果验证策略接口与实际控制运行，任务迁移能力仍需通过成功策略和更多配对场景评估。

- [Lift MuJoCo 报告](../validation/2026-09-29/mujoco_lift/report.json)
- [PickPlace MuJoCo 报告](../validation/2026-09-29/mujoco_pick_place/report.json)

`grasp_v2` 第 50 次迭代的已保存模型也完成相同检查。两项策略导出覆盖四个环境、500 个步骤，实际观测误差为 0，策略动作误差小于 `8.4e-7`，ActionManager 目标误差小于 `3e-7`。MuJoCo 各完成四个 episode，成功率均为 0/4；每项 2000 个姿态的最大位置误差分别约 2.07μm、1.99μm。Lift 的 `grasp_v2` 成功条件同时要求实际双夹爪接触与 0.25 秒保持。

- [grasp_v2 Lift 策略导出](../validation/2026-09-29/grasp_v2_portable_lift/validation.json)
- [grasp_v2 PickPlace 策略导出](../validation/2026-09-29/grasp_v2_portable_pick_place/validation.json)
- [grasp_v2 Lift MuJoCo](../validation/2026-09-29/grasp_v2_mujoco_lift/report.json)
- [grasp_v2 PickPlace MuJoCo](../validation/2026-09-29/grasp_v2_mujoco_pick_place/report.json)

## 相同动作的动力学比较

当前优先推进 sim2sim，真机工作暂缓，RL 训练保持停止。验证使用已保存的策略与实际 Isaac 动作记录。

`sim2sim compare` 将 Isaac 验证轨迹中每步已检查的关节目标发送给 MuJoCo，初始关节位置与速度保持一致，比较每个控制步骤之前的关节、物体与夹爪接触数据。源记录发生 episode 终止时结束对应连续片段。该入口使用记录动作，不执行策略反馈计算。

```bash
openso101 sim2sim compare --policy outputs/rl_progress/lift_grasp_v2_portable_50 \
  --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
  --episodes 4 --steps 500 --output outputs/lift_dynamics_comparison
```

`comparison.hdf5` 保存两个模拟器的状态和同一关节目标；`report.json` 保存每个关节的 RMSE、最大位置差异、峰值时刻、速度、接触力，以及源轨迹、模型、代码和输出 SHA256。已有输出、无效数量、超过源记录的数量及缺少实际 PD 的请求立即终止。

`01f075e` 对 grasp_v2 第 50 次迭代模型的实际轨迹完成以下检查：

| 指标 | Lift | PickPlace |
|---|---:|---:|
| 配对环境 | 4 | 4 |
| 每环境连续控制步骤 | 250 | 400 |
| 初始关节位置、速度误差 | 0 | 0 |
| 最大关节位置差异 | 0.71237 rad | 0.23731 rad |
| 该峰值时刻与关节 | 0.08 s，shoulder_pan | 0.10 s，shoulder_lift |
| Isaac 最大关节速度 | 2.01423 rad/s | 2.01450 rad/s |
| MuJoCo 最大关节速度 | 15.80654 rad/s | 6.43542 rad/s |
| 最大物体位置差异 | 3.80325 mm | 2.89232 mm |

机器人坐标检查仍保持微米级位置误差。动力学差异在控制开始阶段出现，速度行为需要重点验证：Isaac 配置使用 2 rad/s 的 solver 速度限制，当前 MuJoCo 模型没有该限制。Isaac 的评估配置还保留物理随机化，MuJoCo 使用名义 PD、官方 MJCF 惯性和 frictionloss。上述结果给出实际差异，单项原因的影响需要对应参数控制实验。

`rl export` 的 HDF5 逐步保存实际 `joint_stiffness`、`joint_damping`、`joint_armature`、`joint_friction_coeff`、`joint_vel_limits`，以及物体线速度与角速度。包含这些字段的新记录可以使用 `sim2sim compare --recorded-pd`，将每环境实际 PD 参数用于 MuJoCo 原生 actuator，物体初始速度使用实际记录。旧轨迹缺少物体速度时，报告明确记录 MuJoCo 初始物体速度为零。

当前 sim2sim 验证目标是检查相同初始状态与关节目标下的速度、实际 PD、质量、惯性和接触行为，并使用成功策略验证任务迁移。此报告保持 `physics_equivalence_verified=false` 与 `task_success_verified=false`。

两项已保存模型使用新导出入口各完成四环境、500 步实际 Isaac 推理与动作检查，观测重建误差为 0，策略动作误差小于 `8.4e-7`。实际记录显示六关节速度限制均为 2 rad/s，各环境的 PD 参数存在随机化；记录中的 `joint_armature` 与 `joint_friction_coeff` 均为零。

使用同一新轨迹、同一初始状态和完全相同的关节目标，对名义 PD 与记录的实际 PD 分别运行 MuJoCo，结果如下。配对输入、模型、时间步和源文件 SHA256 检查通过。

| 指标 | Lift 名义 PD | Lift 实际 PD | PickPlace 名义 PD | PickPlace 实际 PD |
|---|---:|---:|---:|---:|
| 最大关节位置差异，rad | 0.71237 | 0.63094 | 0.23731 | 0.28084 |
| MuJoCo 最大关节速度，rad/s | 15.80654 | 13.37341 | 6.43542 | 7.03054 |

实际 PD 参数仍保留明显的速度与位置差异。当前重点是速度限制的驱动行为，以及机器人实际质量、惯性和接触参数；MJCF 中保留的 armature 与 frictionloss 需要和 Isaac 的实际参数分别检查。

- [Lift 动力学比较](../validation/2026-09-29/lift_dynamics_comparison_report.json)
- [PickPlace 动力学比较](../validation/2026-09-29/pick_place_dynamics_comparison_report.json)
- [六项输入拒绝检查](../validation/2026-09-29/comparison_guards_report.json)
- [Lift 新推理记录检查](../validation/2026-09-29/lift_physics_export_report.json)
- [PickPlace 新推理记录检查](../validation/2026-09-29/pick_place_physics_export_report.json)
- [Lift 名义 PD 比较](../validation/2026-09-29/lift_physics_nominal_report.json)
- [Lift 实际 PD 比较](../validation/2026-09-29/lift_recorded_pd_report.json)
- [PickPlace 名义 PD 比较](../validation/2026-09-29/pick_place_physics_nominal_report.json)
- [PickPlace 实际 PD 比较](../validation/2026-09-29/pick_place_recorded_pd_report.json)
- [实际输入与来源配对检查](../validation/2026-09-29/pd_comparison_pairs_report.json)

## 原生 Isaac 速度限制实验

`scripts/check_isaac_velocity_limit.py` 在同一实际 Isaac 场景中建立四组配对环境，每组使用相同的关节与物体初始状态、实际 PD 和已记录的绝对关节目标。机器人与物体的质量、惯性、材质，以及关节参数通过原生接口读取并检查配对一致性。速度限制通过 `write_joint_velocity_limit_to_sim` 写入 PhysX，再读取实际 solver 参数确认；每组分别使用源记录的 2 rad/s 和实验参数 1000 rad/s。

```bash
PYTHONPATH=src /home/jixin/workspace/envs/openso101-v2/bin/python \
  scripts/check_isaac_velocity_limit.py \
  outputs/rl_progress/lift_physics_portable_50 \
  outputs/rl_progress/lift_velocity_limit --steps 100
```

在配置好的 Isaac 环境中执行，PickPlace 使用 `pick_place_physics_portable_50` 和独立输出目录。每个任务使用八个环境、100 个控制步骤，控制周期 0.02 秒，物理周期 0.01 秒。实验直接执行物理步骤，期间没有任务重置和策略反馈计算。输出 `velocity_limit.hdf5` 与 `report.json`，保存实际轨迹、输入、物理参数，以及代码和源文件 SHA256。

`51647ee` 在 `jd_B300` 完成两个任务的实际运行：

| 指标 | Lift | PickPlace |
|---|---:|---:|
| 配对初始关节位置、速度误差 | 0 | 0 |
| 2 rad/s 限制下最大关节速度 | 2.01409 rad/s | 2.01432 rad/s |
| 1000 rad/s 限制下最大关节速度 | 19.46912 rad/s | 9.69941 rad/s |
| 配对轨迹最大关节位置差异 | 0.54092 rad | 0.26580 rad |
| 2 rad/s 轨迹与源记录最大关节位置差异 | 0.000253 rad | 0.000257 rad |

该实验确认速度限制会明显改变这些动作的实际运动。每组的物理参数和 PD 一致性检查通过；新场景的随机物理参数没有完整恢复为源记录参数，因此 `source_physics_reproduction_verified=false`。1000 rad/s 仅用于本次因素控制实验。跨模拟器物理等价与任务成功仍未验证。

下一项 sim2sim 目标是在 MuJoCo 中实现并验证速度受限的驱动行为，比较相同输入下的速度与关节轨迹，同时记录机器人质量、惯性和接触参数。该驱动需要保持动力学与接触求解的作用；直接修改关节速度不能提供对应的 solver 行为验证。

六项实际无效输入检查均在启动 Isaac 前终止，覆盖单步骤、无效或非有限速度限制、源轨迹数量不足、episode 边界和已有输出目录。

- [Lift 原生速度限制报告](../validation/2026-09-29/lift_native_velocity_report.json)
- [PickPlace 原生速度限制报告](../validation/2026-09-29/pick_place_native_velocity_report.json)
- [六项输入拒绝检查](../validation/2026-09-29/native_velocity_guards_report.json)

## sim2real 视觉策略检查

`PortablePolicy` 接收状态观测，其中物体位置、任务目标与抓取状态需要实际观测来源。现有真机 deploy 使用视觉 student 或 LeRobot policy。视觉 student 与 PortablePolicy 共用关节动作转换函数，随后按已有 SO-101 映射转换为 LeRobot motor units。

```bash
openso101 sim2real validate --policy-path outputs/verify_student_ready \
  --episode outputs/runtime_dataset/episodes/episode_000000.hdf5 \
  --output outputs/student_check --device cpu
```

入口校验 HDF5、任务、场景 hash、关节顺序及 student 文件 hash，对全部实际 RGB 与关节观测执行预处理、推理和动作转换，保存六关节 motor commands 及报告。该检查不连接机器人。

本次使用已有自定义场景的 12 帧双相机数据和已保存的 student，全部帧推理通过；训练控制周期为 1/60 秒，报告要求部署频率为 60Hz。该 student 的训练与任务成功仍需验证。硬件校准、实际相机、停止控制和机器人任务完成使用真机验收，报告保持 `hardware_run_verified=false`。

[sim2real student 检查报告](../validation/2026-09-29/sim2real_student/report.json)

## sim2real 相机与停止控制

`camera-check` 使用部署入口相同的 LeRobot OpenCVCamera 与图像转换函数，独立检查双相机尺寸、FPS、RGB 格式和实际读取帧数：

```bash
openso101 sim2real camera-check \
  --wrist-camera-path /dev/video0 --overhead-camera-path /dev/video2 \
  --camera-width 128 --camera-height 128 --fps 60 --frames 60 \
  --output outputs/camera_check.json
```

设备路径优先于 `--wrist-camera-index` 和 `--overhead-camera-index`。此入口同时接受已有视频文件，从首帧开始读取并校验文件的尺寸与 FPS。文件编码需要当前 OpenCV 的解码支持；本次检查使用 H.264 MP4。设备输入执行 LeRobot warmup，并请求指定的采集参数。图像转换要求 `uint8 H×W×3 RGB`，输出归一化的 `(1, 3, H, W)` tensor，格式错误立即终止。

部署入口将相机 FPS 设置为 `--fps`，在连接机器人之前检查正数配置、独立相机来源、student 控制周期以及 preprocessor／postprocessor。机器人部署需要实际相机设备；已有视频通过 `camera-check` 检查。共享 canonical 初始姿态可以独立导入，生成动作时执行相同的 motor-unit 转换与范围限制。

```bash
openso101 sim2real deploy --policy-path outputs/verify_student_ready \
  --follower-port /dev/ttyACM0 --follower-id so101 \
  --wrist-camera-path /dev/video0 --overhead-camera-path /dev/video2 \
  --camera-width 128 --camera-height 128 --fps 60 \
  --stop-file outputs/deploy.stop --max-steps 60 --device cpu
```

部署前需要使用该 follower_id 对应的实际 LeRobot 校准，并核查相机设备和动作转换。当前示例中的设备路径需要根据连接主机设置，真机信息仍待提供。

`--stop-file` 存在时，启动入口直接退出。运行期间在初始化动作发送之前、每次观测读取之前和每次策略动作发送之前检查该文件；可通过另一个终端执行 `touch outputs/deploy.stop` 请求停止。退出调用 LeRobot `follower.disconnect()`，默认关闭电机扭矩，并关闭双相机。Ctrl+C 同样执行连接清理。文件检查的响应时间受同步设备读取和推理耗时影响；真机扭矩、退出行为和紧急停止需要实际设备验收。

`9d1d711` 的实际验证包括：双相机各 12 帧 H.264 文件读取、128×128 和 60 FPS metadata、部署图像转换、保存的 student 全部 12 帧推理、停止文件与控制频率预检查，以及 44 项 CPU 检查。预检查使用真实 student 和文件，通过函数调用观察确认三个检查均未连接 follower；初始姿态生成没有导入 Isaac。源视频、编码转换误差和文件 hashes 保存在独立报告中。此次使用录制文件，`hardware_run_verified=false`，真机采集、控制与任务成功仍需验收。

- [双相机读取报告](../validation/2026-09-29/camera_read_report.json)
- [双相机编码转换报告](../validation/2026-09-29/camera_transcode_report.json)
- [部署预检查报告](../validation/2026-09-29/deploy_preflight_report.json)
