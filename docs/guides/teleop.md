# LeRobot SO101 Teleoperation

This document captures the current working teleoperation setup for OpenSO-101.
It is the handoff point for real SO101 leader-arm teleop, local HDF5 collection,
and later LeRobot dataset/policy workflows.

## Current Architecture

- OpenSO-101 remains the simulation source of truth: task registration,
  scene objects, rewards, cameras, and domain-randomization logic stay in this
  repository.
- LeRobot is used at the boundary: it reads the physical SO101 leader arm and
  later consumes exported demonstration datasets.
- The simulated follower uses the canonical SO101 USD asset at
  `assets/so101/usd/SO-ARM101-USD.usd`.
- The canonical robot config is `openso101.robots.SO_ARM101_CFG`.
- Teleop tasks use absolute six-joint position targets:
  `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.
- Training tasks can still use their RL action configs; teleop action semantics
  are intentionally separated from PPO-style normalized actions.

## Robot And Contact Notes

The SO101 USD includes the gripper camera mount geometry and the current
robot visuals/colliders. The wrist sensor is spawned under:

```text
{ENV_REGEX_NS}/Robot/gripper/gripper_cam
```

OpenSO-101 uses the USD's authored colliders verbatim (no custom spawn-time
collision rewrites). The robot config matches
[liorbenhorin/lerobot_so101_teleop](https://github.com/liorbenhorin/lerobot_so101_teleop)
in trusting the upstream asset; combined with compliant low-stiffness PD
gains (e.g. gripper `k=4 / d=0.3`), the jaws pinch the cube reliably without
silently disabled colliders. Earlier custom collision-spawn functions caused
PhysX `MeshMergeCollisionAPI`-vs-standalone-`CollisionAPI` conflicts that
silently dropped gripper collision.

## Cameras

The teleop-vision task records and displays:

- `wrist_camera`: attached to the gripper camera mount in the robot USD.
- `overhead_camera`: fixed external view over the table.

Graphical teleop launches open three views:

- default perspective viewport
- wrist camera viewport
- overhead camera viewport

Use `--no-camera-viewports` only when the viewport UI is not needed.

## Data Collection

Teleop records automatically by default. Episodes are local HDF5 files under
`teleop_data/`; no Hugging Face upload happens during hardware teleop.

Each saved episode includes:

- `observations/qpos`
- `observations/qvel`
- `action`
- `timestamps`
- `observations/images/wrist_camera`
- `observations/images/overhead_camera`

The file also carries metadata such as semantic LeRobot joint names, simulated
joint names, task text, and success/cancel state.

## Launch Command

Use the `openso101` conda environment:

```bash
conda run -n openso101 openso101 il record \
  --task OpenSO101-PickPlace-v0 \
  --leader-port /dev/ttyACM0 \
  --leader-id leader_arm_1 \
  --profile-teleop
```

The default task target is the canonical PickPlace gym ID with teleop semantics
and cameras enabled via kwargs:

```python
gym.make("OpenSO101-PickPlace-v0", action_mode="teleop", cameras=True)
```

The teleop object is the same shared 3 cm Isaac Lab `CuboidCfg` used by RL
tasks. Prebuilt Isaac block USDs are not exposed in the teleop command because
their mesh/body layout did not behave reliably with the SO101 gripper.

## Keyboard Controls

### 键盘驱动机器人

图形界面启动：

```bash
openso101 il record --task OpenSO101-PickPlace-v0 \
  --teleop-device keyboard --keyboard-input window \
  --repo-root outputs/keyboard_pick_place
```

SSH 终端使用 `ssh -tt jd_B300` 进入已配置的仿真环境，执行：

```bash
openso101 il record --task OpenSO101-PickPlace-v0 \
  --teleop-device keyboard --keyboard-input terminal --headless \
  --no-camera-viewports --repo-root outputs/keyboard_pick_place
```

键盘控制使用机器人实际 Jacobian，执行关节限位与速度限制，并按照仿真控制周期处理输入。释放方向按键后保持最近的关节目标，恢复 checkpoint 时更新控制参考。终端使用 prompt_toolkit 读取按键；方向按键在最后一次输入后 150ms 释放，持续按住按键通过终端重复输入继续移动。

录制 FPS 根据 `sim.dt * decimation` 计算。当前 LeRobot 导出要求整数 FPS；指定 `--fps` 时必须与实际控制频率相同。频率与录制处理位于 `teleop/timing.py`，采集和回放状态位于 `teleop/sim_state.py`。

| 按键 | 操作 |
|---|---|
| ↑ / ↓ | 沿世界坐标的 +x / −x 移动 |
| ← / → | 沿世界坐标的 +y / −y 移动 |
| PageUp / PageDown | 沿 +z / −z 移动 |
| A / D | 控制 yaw |
| Space | 打开夹爪 |
| G | 关闭夹爪；图形窗口也支持 Shift |
| C / R | 保存当前 checkpoint / 恢复 checkpoint |
| S | 人工标记成功并保存 episode |
| Q | 取消当前 episode 并退出 |

终端模式也支持 Ctrl+C 和 Ctrl+D 取消录制。没有交互终端的输入会在启动仿真前终止；图形窗口输入需要实际窗口。PickPlace 的放置检查使用两侧夹爪接触测量、夹爪打开状态、目标位置及持续稳定时间。

### 录制操作

- `S`：人工标记成功，保存当前 episode 并退出。
- `Q`：取消当前 episode 并退出。
- `C`：保存机器人、场景状态与录制位置的 checkpoint。
- `R`：恢复最近的 checkpoint。键盘控制参考同时更新；leader 模式等待实际关节与恢复位置满足同步阈值后继续控制。

任务成功条件满足时，录制默认显示 `[y/N]` 保存提示；`--auto-save` 自动保存满足条件的 episode。

## Export To LeRobot Later

录制 HDF5 后，可以导出本地 LeRobot 数据集：

每个 episode 的首帧确定 `sim` 状态字段，后续帧必须包含相同字段和形状。记录包含三维 `environment_origin`，回放根据源环境与目标环境的原点转换物体与任务目标的 world frame 坐标。PickPlace 记录物体状态、任务阶段、任务目标、物体初始位置和 `command_placement_hold_seconds`；回放时恢复已有记录字段。数据写入和状态恢复中的错误会立即报告并终止。

```bash
openso101 il export \
  --repo-root outputs/my_dataset \
  --repo-id local/my_dataset \
  --output outputs/my_lerobot_dataset
```

默认仅导出成功 episode，跳过每个 episode 开头的五帧，并要求剩余至少十帧。检查控制与记录流程时，可以设置 `--include-failures --skip-leading-frames 0` 保留全部帧。`meta/openso101_export.json` 保存源文件 SHA256、源帧数、导出帧数、成功标记和筛选配置；自定义场景同时保存场景副本与 `meta/scenes.json`。双相机视频、动作和关节观测均包含在导出的数据中。

导出前检查全部源 HDF5、整数 FPS、图像格式、筛选结果与场景文件。异步模式使用单个工作线程保存 episode，并在下一次写入 dataset buffer 前等待保存完成。`--no-async-flush` 使用同步保存。已有输出需要明确设置 `--overwrite-export`，覆盖操作保留原数据目录。同步与异步导出的全部帧检查见 [CPU、LeRobot 导出与打包验证](../validation/2026-10-08/il_pipeline/README.md)。

本地 LeRobot metadata 检查读取实际 info、tasks、episodes、stats 与 Parquet。episode 范围必须连续，帧数与实际 Parquet 一致；动作和状态的六个关节顺序、双相机视频文件、视频 FPS 与有限统计量均需要完整。视频缺少时在 LeRobot 数据读取之前终止。

使用 Hub 上传入口发布数据集：

```bash
conda run -n openso101 openso101 il push \
  --repo-root ./teleop_data/openso101_pickplace_teleop \
  --repo-id ${HF_USER}/openso101_pickplace_teleop \
  --overwrite-export
```

## 回放与验证报告

正式回放入口可以从指定帧恢复记录中的关节和场景状态，并执行指定范围的动作：

```bash
openso101 il replay \
  --episode outputs/my_dataset/episodes/episode_000000.hdf5 \
  --checkpoint-frame 120 --start-frame 120 --stop-frame 180 \
  --hold-steps 0 --report outputs/replay_report.json \
  --headless --no-camera-viewports
```

`--checkpoint-frame` 指定恢复帧；`--start-frame` 与 `--stop-frame` 指定动作范围，结束帧不包含在范围中。没有指定恢复帧时使用所选 checkpoint，没有 checkpoint 时使用第零帧。`--hold-steps` 默认执行 30 个保持步骤，设置为零可直接执行记录动作。

`--report` 检查控制周期与源 FPS、已记录状态的恢复误差、ActionManager 每步动作、关节有限数值，以及 wrist／overhead RGB 的形状与有限数值。报告保存源文件与检查代码的 SHA256、实际检查帧数和源任务成功标记；状态与动作误差阈值为 `1e-6`。检查覆盖环境索引零；已有报告文件、空动作范围和负数保持步骤会在启动 Isaac 前终止。

`replay_verified` 表示请求范围的程序检查全部通过。`task_success_verified` 读取实际任务 termination；完整物理轨迹的重复性由 `physics_state_reproducibility_verified` 单独记录。旧数据仅检查已保存的状态字段，可通过 `source_sim_fields` 与 `restore_errors` 查看检查范围。

原生策略或 scripted controller 的采集文件保存 `task_profile`、`environment_mode`、`physics_dt`、`reward_discount` 与 `environment_origin`。`grasp_v4` 回放从这些 metadata 恢复原生任务、绝对关节目标和控制周期，读取实际双相机尺寸。原生文件需要包含环境原点。

2026-10-07 完成四环境成功采集到单环境的实际 Lift 回放：243 帧、50 Hz、动作和全部记录状态的恢复误差均为零，实际满足 0.25 秒持物要求。该成功 episode 的 LeRobot 导出也完成两台相机全部 243 帧的读取检查，动作和关节观测转换误差均为零。报告与操作范围见 [采集与回放记录](../validation/2026-10-06/README.md)。

实际运行覆盖 PickPlace 的第 120–179 帧、自定义场景全部 12 帧及旧键盘数据的前 60 帧；三项进程均正常退出，动作和状态恢复误差均为零。报告见 [回放运行记录](../validation/2026-09-29/README.md)。

## Diagnostics

If contact errors appear again:

1. Check stale Isaac processes first:

   ```bash
   nvidia-smi
   ```

   Old `openso101/bin/python` or Isaac processes can keep several GB of GPU
   memory and trigger PhysX allocation failures at first contact.

2. Confirm gripper/jaw collision approximation after spawn:

   ```text
   /World/envs/env_0/Robot/gripper/collisions approximation=convexDecomposition
   /World/envs/env_0/Robot/jaw/collisions approximation=convexDecomposition
   ```

3. Use `--profile-teleop` to inspect leader read time, sim step time, recording
   time, loop time, and joint tracking error.

Known useful validation commands:

```bash
conda run -n openso101 env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_so101_teleop_scene_cfg.py \
  tests/test_lerobot_so101_mapping.py \
  tests/test_hdf5_teleop_recorder.py \
  tests/test_teleop_agent_keyboard.py \
  -q

python3 -m compileall -q src/openso101 tests
git diff --check
```
