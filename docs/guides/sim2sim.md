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
