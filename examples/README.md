# Examples

Copy-paste recipes for the main OpenSO-101 workflows. All assume a working
install (`bash scripts/install.sh`) and the SO-101 USD asset fetched
(`bash scripts/fetch_so101_usd.sh`). See [`docs/guides/`](../docs/guides/) for
the full walkthroughs.

## Reinforcement learning

```bash
# Train PPO on pick-and-place (headless, visual DR on)
openso101 rl train --task OpenSO101-PickPlace-v0 --algo ppo --headless --visual-dr

# Quantitatively evaluate a checkpoint — success rate + grasp/lift/place funnel
openso101 rl eval \
  --task OpenSO101-PickPlace-v0 \
  --checkpoint logs/rsl_rl/pick_place/<run>/model_best.pt --headless

# Watch the best checkpoint roll out
openso101 rl play \
  --task OpenSO101-PickPlace-v0 \
  --checkpoint logs/rsl_rl/pick_place/<run>/model_best.pt

# Plot training curves (reward, success rate, per-term breakdown, losses)
openso101 rl plot --task OpenSO101-PickPlace-v0
```

## Imitation learning (teleop → dataset → train → deploy)

```bash
# 1. Record teleop demos from a real SO-101 leader arm (HDF5)
openso101 il record \
  --task OpenSO101-PickPlace-v0 \
  --leader-port /dev/ttyACM0 --leader-id leader_arm_1 \
  --repo-root teleop_data/openso101_pickplace

# 2. Convert HDF5 → LeRobot dataset (and push to the Hub)
openso101 il push \
  --repo-root teleop_data/openso101_pickplace \
  --repo-id <your-hf-username>/openso101_pickplace

# 3. Train an ACT (or diffusion) policy via LeRobot
openso101 il train --policy act --dataset <your-hf-username>/openso101_pickplace

# 4. Roll out in sim, then measure success rate over many episodes
openso101 il play --task OpenSO101-PickPlace-v0 \
  --policy-path logs/lerobot/openso101_act/<timestamp>/checkpoints/last/pretrained_model
openso101 il eval --task OpenSO101-PickPlace-v0 \
  --policy-path logs/lerobot/openso101_act/<timestamp>/checkpoints/last/pretrained_model --headless

# 5. Deploy the same checkpoint on the physical robot
openso101 sim2real deploy \
  --policy-path logs/lerobot/openso101_act/<timestamp>/checkpoints/last/pretrained_model \
  --follower-port /dev/ttyACM1 --follower-id follower_arm_1 \
  --wrist-camera-index 0 --overhead-camera-index 2
```

## Register a custom task

See [`register_custom_task.py`](register_custom_task.py) for the one-decorator
extension API, and [`docs/guides/add_a_task.md`](../docs/guides/add_a_task.md)
for a full walkthrough.
