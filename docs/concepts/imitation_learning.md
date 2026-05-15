# Imitation Learning

The imitation-learning (IL) pillar takes the SO-101 from "I can move a leader
arm" to "my policy can do this on its own." It spans three layers:

1. **Teleop boundary** — drive a simulated follower from a real SO-101
   leader arm, recording HDF5 frames and metadata.
2. **Dataset** — convert recorded HDF5 to [LeRobot][lerobot] format and
   push to the Hugging Face Hub.
3. **Policy training** — train ACT or Diffusion policies on the dataset
   (sub-project C; not yet wired up).

## Today's Flow: Teleop → HDF5 → LeRobot Hub

This portion is **fully implemented** and works end-to-end.

```bash
# 1. Plug in the SO-101 leader arm and find its serial port (Linux: ls /dev/ttyACM*)
openso101 il record \
  --task OpenSO101-PickPlace-v0 \
  --leader-port /dev/ttyACM0 \
  --leader-id leader_arm_1 \
  --repo-root teleop_data/openso101_pickplace
```

The recorder writes per-episode HDF5 files with three layers of state:

| Group | What's in it |
|---|---|
| `/data/action` | Six joint commands at every frame (the leader's intent) |
| `/data/qpos` | Six follower joint positions (the sim's response) |
| `/data/qvel` | Six follower joint velocities |
| `/data/cameras/<name>/rgb` | Per-camera RGB frames if cameras are enabled |
| `/sim_state` | Object pose, gripper width, end-effector pose, success/failure flags |

Episodes can be saved or cancelled with `s` / `c` keystrokes during teleop.

Once you have a few good episodes:

```bash
openso101 il push \
  --repo-root teleop_data/openso101_pickplace \
  --repo-id <username>/openso101-pickplace
```

This converts each HDF5 episode to LeRobot's parquet+video format and uploads
to the Hub. The local `teleop_data/...` directory becomes a fully-functional
LeRobot dataset; you can also pass `--dry-run` to see the upload plan
without pushing.

## Replay

To sanity-check a recorded episode in sim:

```bash
openso101 il replay --episode teleop_data/openso101_pickplace/episode_000.hdf5
```

Useful flags:
- `--list-checkpoints` — print the success/keypress checkpoints in the file
- `--start-frame N --stop-frame M` — slice the playback range
- `--real-time` — wall-clock pace instead of as-fast-as-possible

## Tomorrow's Flow: Dataset → Policy

This is **sub-project C** and is not yet implemented. The intended CLI:

```bash
openso101 il train --policy act --dataset <username>/openso101-pickplace
openso101 il play  --task OpenSO101-PickPlace-v0 --policy-path runs/act/best.pt
```

The training stack will live under `openso101/il/`:
- `il/policies/act.py` — [Action Chunking Transformer][act]
- `il/policies/diffusion.py` — [Diffusion Policy][diffusion]
- `il/datasets/lerobot_adapter.py` — wraps a LeRobot dataset for PyTorch
- `il/runners/trainer.py` — training loop

## Bridge to Sub-Project F

The IL pillar consumes data; sub-project F (synthetic data generation)
produces it. A small set of teleop demonstrations can be amplified into
hundreds or thousands of trajectories through [MimicGen][mimicgen] or
[Isaac Lab Mimic][isaaclab-mimic], then fed straight into the IL trainer.
See [`data_generation.md`](data_generation.md).

[lerobot]: https://github.com/huggingface/lerobot
[act]: https://github.com/tonyzhaozh/act
[diffusion]: https://github.com/real-stanford/diffusion_policy
[mimicgen]: https://github.com/NVlabs/mimicgen
[isaaclab-mimic]: https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_mimic
