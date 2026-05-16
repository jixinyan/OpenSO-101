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

The recorder writes per-episode HDF5 files (one file per episode) with:

| Dataset / group | What's in it |
|---|---|
| `action` | Six joint commands at every frame (the leader's intent) |
| `observations/qpos` | Six follower joint positions (the sim's response) |
| `observations/qvel` | Six follower joint velocities |
| `observations/images/<camera>` | Per-camera RGB frames (e.g. `wrist_camera`, `overhead_camera`) |
| `sim/...` | Optional sim state (object pose, command stage, goal positions) for replay |
| `checkpoints/frame_index` | Frames marked with the `C` key |

Top-level attrs include `format`, `dataset_id`, `task`, `fps`,
`success`, `joint_names`, `sim_joint_names`, `lerobot_action_names`,
and `camera_names`. See
`openso101.teleop.hdf5_recorder.validate_hdf5_episode` for the
authoritative schema check.

During teleop the four-key map is:
- `S` — mark the current episode SUCCESS, save it, exit.
- `Q` — cancel the current episode and exit (discard data).
- `C` — checkpoint the current frame.
- `R` — restore robot + env state to the most recent checkpoint.

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

## Dataset → Policy

`openso101 il train` is a thin wrapper around `lerobot.scripts.train`
so the IL ecosystem (ACT, Diffusion, VQ-BeT) is available without us
maintaining a parallel trainer. The local LeRobot dataset produced by
`il push` is the input; checkpoints land under
`logs/lerobot/openso101_<policy>/<timestamp>/`.

```bash
openso101 il train --policy act --dataset teleop_data/openso101_pickplace/lerobot_dataset
openso101 il play  --task OpenSO101-PickPlace-v0 \
                   --policy-path logs/lerobot/openso101_act/<timestamp>
```

`il play` rolls the trained checkpoint out in the sim env with rewards
+ terminations active (so the operator can read success signals). Pass
`--action-mode teleop` to switch to the long-episode no-rewards variant
matching `il record`. The same checkpoint deploys on the real follower
via `openso101 sim2real deploy --policy-path <ckpt> --follower-port ...`.
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
