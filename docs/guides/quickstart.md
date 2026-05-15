# Quickstart

Get from a fresh OpenSO-101 install to a trained PPO checkpoint in
under 20 minutes.

> Assumes you've completed [`install.md`](install.md) and can run
> `openso101 envs list` successfully.

## 1. Smoke Test the Sim (1 min)

Open `OpenSO101-PickPlace-v0` with a random policy:

```bash
openso101 envs random --task OpenSO101-PickPlace-v0 --steps 200
```

Expected: an Isaac Sim window opens, you see the SO-101 flailing
randomly above a green cube on a table, and the script exits.
This confirms Isaac Lab + the OpenSO-101 task registration both work.

## 2. Train PPO (10–15 min on RTX 4080)

```bash
openso101 rl train \
  --task OpenSO101-PickPlace-v0 \
  --algo ppo \
  --headless \
  --max_iterations 1500
```

Watch the W&B dashboard (`Episode/Reward/total`) climb from −2.0 to
+8.0 over ~1500 iterations.

Output logs land under `logs/rsl_rl/openso101_pickplace/<timestamp>/`.
The runner saves both `model_<iter>.pt` (every N iters) and
`model_best.pt` (whenever 100-episode mean reward hits a new high).

## 3. Replay the Best Checkpoint (1 min)

```bash
openso101 rl play \
  --task OpenSO101-PickPlace-v0 \
  --checkpoint logs/rsl_rl/openso101_pickplace/<run>/model_best.pt
```

Isaac Sim opens and runs the trained policy. You should see the arm
pinch the cube, lift, and drop it on the goal marker.

## 4. (Optional) Record a Teleop Demo

If you have a real SO-101 leader arm plugged in:

```bash
openso101 il record \
  --task OpenSO101-PickPlace-v0 \
  --leader-port /dev/ttyACM0 \
  --leader-id leader_arm_1 \
  --repo-root teleop_data/my_first_demo
```

- Move the leader arm to teleoperate the simulated follower.
- Press `s` to save the episode, `c` to cancel, `q` to quit.

## 5. (Optional) Push the Dataset

```bash
openso101 il push \
  --repo-root teleop_data/my_first_demo \
  --repo-id <your-hf-username>/my-first-so101-demo
```

The dataset will appear at `https://huggingface.co/datasets/<your-hf-username>/my-first-so101-demo`.

## Next Steps

- [Tasks & envs concepts](../concepts/tasks_and_envs.md) — how the framework is organized.
- [Add a custom task](add_a_task.md) — write your own.
- [RL algorithms](../concepts/rl_algorithms.md) — what's supported and what's planned.
- [Teleop guide](teleop.md) — full teleop setup and dataset workflow.

## CLI Cheat Sheet

```bash
# Discovery
openso101 envs list
openso101 envs preview --task OpenSO101-PickPlace-v0   # with cameras

# RL
openso101 rl train --task <gym-id> --algo ppo [--headless --max_iterations N]
openso101 rl play  --task <gym-id> --checkpoint <path>
openso101 rl plot  --task pick_place [--save]

# IL
openso101 il record --task <gym-id> --leader-port <port> --leader-id <id> --repo-root <dir>
openso101 il push   --repo-root <dir> --repo-id <hf-id>
openso101 il replay --episode <hdf5-path>
```
