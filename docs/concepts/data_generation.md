# Synthetic Data Generation

> **Status:** sub-project F. The framework hooks exist; the generators
> themselves are skeletons.

A handful of human demonstrations is rarely enough to train a robust IL
policy. **Synthetic data generation** (SDG) turns each demo into many
trajectories by replaying it under controlled perturbations (different
object poses, lighting, noise levels) and keeping only the successful
rollouts. OpenSO-101 ships adapters for two SDG stacks:

- **[MimicGen][mimicgen]** — NVIDIA Research's open-source data
  generation framework, originally targeting Robosuite and now usable
  with arbitrary Isaac Lab tasks.
- **[Isaac Lab Mimic][isaaclab-mimic]** — Isaac Lab's first-party
  successor / wrapper around MimicGen.

## CLI

```bash
openso101 data generate \
  --backend mimicgen \
  --task OpenSO101-PickPlace-v0 \
  --seed-dataset <username>/openso101-pickplace \
  --num-trials 1000 \
  --output-dir teleop_data/openso101_pickplace_synthetic
```

Today this exits `2` with a "sub-project F" notice. When the bodies land,
the command will:

1. Load the seed LeRobot dataset.
2. Spawn the task env with domain randomization (per `sim2real` config).
3. Replay each seed episode under N perturbations.
4. Keep only successful rollouts (using the task's `success` termination).
5. Export the result as a fresh LeRobot dataset on disk.

## Backends

| Backend | When to use it |
|---|---|
| `mimicgen` | More mature; richer perturbation parameterization; non-Isaac-Lab tasks. |
| `isaaclab_mimic` | First-party Isaac Lab integration; fewer setup steps; recommended for OpenSO-101 tasks. |

## How It Plugs Into the Stack

```
Real teleop demos (sub-project IL)
  ↓ openso101 il record
HDF5 + LeRobot dataset
  ↓ openso101 data generate --backend mimicgen
Augmented LeRobot dataset (1000× larger)
  ↓ openso101 il train --policy act
Trained IL policy
  ↓ openso101 il play / openso101 sim2real deploy
On-robot behavior
```

The IL pillar consumes the dataset SDG produces; SDG consumes the small
real dataset IL teleop produces. The two pillars are designed to
short-circuit each other: if your task has a sim-only reset path
(e.g. randomized cube spawn), SDG can run without ever needing a teleop
seed; conversely if your task generalizes from 10 real demos, you can
skip SDG entirely.

## Future Work

When sub-project F lands, the adapter surface at
`openso101/data_gen/{mimicgen,isaaclab_mimic}/generator.py` will expose:

```python
class MimicGenGenerator:
    def __init__(self, task_id: str, seed_dataset: str, ...): ...
    def generate(self, num_trials: int, perturbations: PerturbationCfg) -> Path:
        """Run N trials and return the path to the augmented dataset."""
```

with a matching `IsaacLabMimicGenerator` skin over the same interface.

[mimicgen]: https://github.com/NVlabs/mimicgen
[isaaclab-mimic]: https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_mimic
