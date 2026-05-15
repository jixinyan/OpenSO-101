# LeRobot SO101 Teleoperation

> _Ported from `safe_sim2real/docs/lerobot_so101_teleop.md` during the OpenSO-101 v0.1.0 refactor. Code references updated; historical narrative preserved._

This document captures the current working teleoperation setup for OpenSO-101.
It is the handoff point for real SO101 leader-arm teleop, local HDF5 collection,
and later LeRobot dataset/policy workflows.

## Current Architecture

- OpenSO-101 remains the simulation source of truth: task registration,
  scene objects, rewards, cameras, and domain-randomization logic stay in this
  repository.
- LeRobot is used at the boundary: it reads the physical SO101 leader arm and
  later consumes exported demonstration datasets.
- The simulated follower uses the canonical SO101 USD asset copied under
  `outputs/third_party/so101_usd/SO-ARM101-USD.usd`.
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

The source asset authors the gripper and jaw collision groups as SDF contacts.
On the current Isaac Sim 5.1 / RTX 5070 Laptop setup, that path caused GPU PhysX
narrowphase failures when the gripper touched the cube. OpenSO-101 keeps the
same USD model, but `spawn_so101_usd_with_safe_collisions()` rewrites only:

```text
/Robot/gripper/collisions
/Robot/jaw/collisions
```

from `sdf` to `convexDecomposition` at spawn time. This avoids the failing GPU
SDF contact path while preserving the robot model and wrist camera
mount.

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

- `C`: checkpoint the current episode in memory.
- `R`: resume from the most recent checkpoint of the current episode.
- `Q`: quit and cancel the active episode without saving it.
- `S`: save the current episode.

Checkpoint resume restores simulator state and the recording buffer; it does not
resync from the current physical leader pose.

## Export To LeRobot Later

After recording local HDF5 episodes, export/push as a separate step:

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
