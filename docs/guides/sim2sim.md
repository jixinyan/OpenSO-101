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
