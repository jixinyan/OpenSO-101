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

- `S`: mark the current episode SUCCESS, save it, and exit.
- `Q`: cancel the active episode and exit (data discarded).
- `C`: checkpoint the current frame (robot pose + env state + recording
  position).
- `R`: restore robot pose + env state to the most recent checkpoint.
  The sim snaps back to the checkpoint pose; the leader takes over on
  the next frame (no leader-pose sync required — by design).

The auto-detected goal-success path (run with `--goal-region`) prompts
`[y/N]` for save by default; pass `--auto-save` to commit without the
prompt for unattended batch capture.

## Export To LeRobot Later

录制 HDF5 后，可以导出本地 LeRobot 数据集：

```bash
openso101 il export \
  --repo-root outputs/my_dataset \
  --repo-id local/my_dataset \
  --output outputs/my_lerobot_dataset
```

默认仅导出成功 episode，跳过每个 episode 开头的五帧，并要求剩余至少十帧。检查控制与记录流程时，可以设置 `--include-failures --skip-leading-frames 0` 保留全部帧。`meta/openso101_export.json` 保存源文件 SHA256、源帧数、导出帧数、成功标记和筛选配置；自定义场景同时保存场景副本与 `meta/scenes.json`。双相机视频、动作和关节观测均包含在导出的数据中。

使用 Hub 上传入口发布数据集：

```bash
conda run -n openso101 openso101 il push \
  --repo-root ./teleop_data/openso101_pickplace_teleop \
  --repo-id ${HF_USER}/openso101_pickplace_teleop \
  --overwrite-export
```

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
