<!--
*** OpenSO-101 — README modelled on https://github.com/othneildrew/Best-README-Template
*** GitHub-flavored markdown. Anchors are kebab-case derived from section titles.
-->

<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/KevinYan-831/OpenSO-101">
    <img src="media/logo.png" alt="OpenSO-101 logo" width="280">
  </a>

  <h3 align="center">OpenSO-101</h3>

  <p align="center">
    Open-source robot learning framework for the LeRobot SO-101 in Isaac Lab.
    <br />
    <a href="docs/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="examples/">View examples</a>
    &middot;
    <a href="https://github.com/KevinYan-831/OpenSO-101/issues/new?labels=bug">Report bug</a>
    &middot;
    <a href="https://github.com/KevinYan-831/OpenSO-101/issues/new?labels=enhancement">Request feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
## Table of Contents

- [About the Project](#about-the-project)
  - [Built With](#built-with)
  - [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)


<!-- ABOUT THE PROJECT -->
## About the Project

OpenSO-101 is a single-package research framework for the [LeRobot SO-101][so101-url] 6-DoF arm built on [NVIDIA Isaac Lab][isaaclab-url]. It bundles the four pillars of modern robot learning behind one CLI and one Python API:

1. **Reinforcement Learning** — PPO via [`rsl_rl`][rsl-rl-url] with a `BestCheckpointRunner` that snapshots the best mean-reward checkpoint and a `--visual-dr` flag that shares the same lighting / colour randomization the IL pipeline uses.
2. **Imitation Learning** — leader-arm teleop with async polling, streaming HDF5 recording, [LeRobot dataset][lerobot-url] conversion, and training via the official `lerobot.scripts.train` CLI (ACT, Diffusion, or any policy LeRobot ships). The same checkpoint produced by `openso101 il train` plays back in sim (`openso101 il play`) and deploys on hardware (`openso101 sim2real deploy`). Everything is also exposed as Python functions under `openso101.il` — `load_lerobot_dataset`, `train_il_policy`, `load_policy`, `ACTPolicy`, `DiffusionPolicy` — so notebooks and sweep drivers don't need to shell out.
3. **Sim-to-Real Robustness** — visual, observation, and physics domain randomization shared across all three built-in tasks; a real-arm deploy bridge that drives the Feetech follower via LeRobot's `SO101Follower` while streaming OpenCV camera frames into the policy.
4. **Synthetic Data Generation** *(deferred)* — [MimicGen][mimicgen-url] and [Isaac Lab Mimic][isaaclab-mimic-url] CLI surface is wired but the generator bodies are intentionally not implemented until the RL + IL pipelines are fully validated on human teleop data.

The project is organized so that a researcher can clone, install, and reach a working `openso101 envs list` in well under an hour — and so a downstream contributor can register a custom task with one decorator. Each pillar exposes a stable CLI verb and a stable Python entry point; swapping in a custom algorithm or task does not require forking the framework.

Out of scope (lives in the legacy [`safe_sim2real`][safe-sim2real-url] repository):
- Safe-RL (PPO-Lagrangian, CPO, FOCOPS).
- SB3 / StableBaselines3 wrappers.
- Off-policy RL (SAC).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![Isaac Lab][isaaclab-shield]][isaaclab-url]
[![LeRobot][lerobot-shield]][lerobot-url]
[![rsl_rl][rsl-rl-shield]][rsl-rl-url]
[![Gymnasium][gymnasium-shield]][gymnasium-url]

### Repository Layout

```
OpenSO-101/
├── src/openso101/
│   ├── cli/                  # `openso101 {envs,rl,il,data,sim2real}` dispatch
│   ├── envs/                 # OpenSO101EnvCfg base + register_task decorator
│   ├── robots/so101/         # SO-101 ArticulationCfg, USD spawn, cameras, pose constants
│   ├── tasks/                # Built-in tasks: lift, pick_place, stack (+ shared/)
│   ├── teleop/               # LeRobot leader-arm → simulated follower (async daemon poll)
│   ├── rl/                   # rsl_rl-backed PPO + BestCheckpointRunner
│   ├── il/
│   │   ├── policies/         # ACTPolicy, DiffusionPolicy, load_policy (LeRobot wrappers)
│   │   ├── runners/          # train_il_policy() — programmatic LeRobot trainer
│   │   └── datasets/         # load_lerobot_dataset() — Hub id OR local recorder dir
│   ├── data_gen/             # Synthetic data CLI surface (generator bodies deferred)
│   └── sim2real/
│       ├── domain_randomization/  # visual / observation / physics DR (shared across tasks)
│       └── deploy.py              # real-arm deploy bridge (LeRobot SO101Follower)
├── scripts/
│   └── install.sh            # uv-based installer (resolves isaaclab/lerobot conflict)
├── docs/                     # Concepts, guides, dev diary, this README's siblings
├── examples/                 # End-to-end recipe scripts
├── tests/                    # pytest suite (43 tests, runs without Isaac Sim)
├── constraints.txt           # `setuptools<81` for flatdict's legacy sdist
├── requirements-cuda.txt     # torch cu128 wheels (Blackwell-compatible)
└── pyproject.toml            # `openso101` console_scripts + [tool.uv] resolver overrides
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- **Hardware:** an NVIDIA GPU (RTX 20-series or newer; CUDA 12.x). 6 GB+ VRAM recommended for play, 16 GB+ for training.
- **OS:** Ubuntu 22.04 LTS (Isaac Sim 4.5 / 5.1 official target).
- **Driver:** NVIDIA driver ≥ 560.
- **Python:** 3.11 (Isaac Lab is pinned to 3.11).
- **Conda or Mambaforge:** required to keep Isaac Sim's dependency graph isolated.

### Installation

OpenSO-101 ships a single install wrapper that produces a fully-resolved environment in one command:

```bash
# 1. Clone
git clone https://github.com/KevinYan-831/OpenSO-101.git
cd OpenSO-101

# 2. Create the conda env (Python 3.11 only — heavy deps go in next)
conda env create -f environment.yml
conda activate openso101

# 3. Run the installer (bootstraps uv, installs cu128 torch, then openso101+isaaclab+lerobot)
bash scripts/install.sh

# 4. Verify
openso101 envs list
# OpenSO101-Lift-v0
# OpenSO101-PickPlace-v0
# OpenSO101-Stack-v0
```

**Why `bash scripts/install.sh` instead of `pip install -e .`?** `isaaclab_rl` (pulled in by `isaaclab[all,isaacsim]==2.3.0`) declares `packaging<24`, while `lerobot==0.4.0` declares `packaging>=24.2`. These ranges do not intersect, so pip's resolver hits `ResolutionImpossible`. The `<24` cap is an erroneous upper bound — both packages only call `packaging.version.Version`, which has been API-stable since v21.x. We solve it the honest way: a `[tool.uv] override-dependencies` block in `pyproject.toml` lets [`uv`](https://github.com/astral-sh/uv) rewrite the conflicting metadata at solve time, and the install succeeds in a single resolver pass — no `--no-deps`, no manual fallback dep list. We track the upstream fix at [isaac-sim/IsaacLab#5084](https://github.com/isaac-sim/IsaacLab/issues/5084). See [`docs/guides/install.md`](docs/guides/install.md) for the long-form rationale and troubleshooting.

After a `git pull` you only need:

```bash
bash scripts/install.sh --quick    # `pip install -e . --no-deps`
```

### Quickstart

**Train PPO on PickPlace** (headless, single GPU, visual DR on):

```bash
openso101 rl train --task OpenSO101-PickPlace-v0 --algo ppo --headless --visual-dr
```

**Replay the best checkpoint:**

```bash
openso101 rl play --task OpenSO101-PickPlace-v0 \
  --checkpoint logs/rsl_rl/pick_place/<run-dir>/model_best.pt
```

**Record a teleop demonstration** with a real SO-101 leader arm:

```bash
openso101 il record \
  --task OpenSO101-PickPlace-v0 \
  --leader-port /dev/ttyACM0 \
  --leader-id leader_arm_1 \
  --repo-root teleop_data/openso101_pickplace
```

Recording keys: `S` = save success + exit · `Q` = discard + exit · `C` = checkpoint · `R` = hard-restore to checkpoint. The leader is polled by a daemon thread at ~1 kHz; the simulation reads the latest cached value, so teleop stays smooth even when the sim step takes longer than the bus round-trip.

**Train an IL policy via LeRobot** (delegates to `lerobot.scripts.train`):

```bash
openso101 il train --policy act --dataset teleop_data/openso101_pickplace
# or
openso101 il train --policy diffusion --dataset teleop_data/openso101_pickplace
```

**Play the IL checkpoint in sim:**

```bash
openso101 il play --task OpenSO101-PickPlace-v0 \
  --policy-path outputs/train/<run-dir>/checkpoints/last/pretrained_model
```

**Deploy the same checkpoint on the real robot:**

```bash
openso101 sim2real deploy \
  --policy-path outputs/train/<run-dir>/checkpoints/last/pretrained_model \
  --follower-port /dev/ttyACM1 \
  --follower-id follower_arm_1 \
  --wrist-camera-index 0 --overhead-camera-index 2
```

The motor-unit action space (`[-100, 100]` per joint) is identical in sim and on hardware, so the LeRobot checkpoint loads unchanged in either context. Both `openso101 il play` and `openso101 sim2real deploy` go through the same `openso101.il.policies.load_policy` loader.

**Or drive everything from Python** (notebooks, sweeps, custom tooling):

```python
from openso101.il import (
    load_lerobot_dataset, summarize_lerobot_dataset,
    train_il_policy, load_policy,
)

ds = load_lerobot_dataset("teleop_data/openso101_pickplace")
summarize_lerobot_dataset(ds)

result = train_il_policy(policy="act", dataset=ds.root, steps=200_000)
assert result.succeeded

policy = load_policy(result.last_checkpoint, device="cuda")
# policy.select_action(observation) in either sim or real
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE -->
## Usage

The full CLI surface:

| Group | Verb | What it does |
|---|---|---|
| `envs` | `list` | Print registered OpenSO-101 gym IDs |
|        | `random` | N random-action steps as a smoke test |
|        | `zero` | N zero-action steps |
|        | `preview` | Spawn the env with cameras enabled |
| `rl` | `train` | Train PPO on a task (`--visual-dr` to enable lighting + colour DR) |
|      | `play` | Replay an RL checkpoint |
|      | `plot` | Plot training curves from a run dir |
| `il` | `record` | Record teleop demos to HDF5 + LeRobot, with async leader polling |
|      | `push` | Push a LeRobot dataset to the Hugging Face Hub |
|      | `train` | Shell out to `lerobot.scripts.train` (ACT, Diffusion, ...) |
|      | `play` | Load a LeRobot checkpoint and roll it out in sim |
|      | `replay` | Replay a recorded teleop episode |
| `sim2real` | `deploy` | Drive the real SO-101 from a LeRobot checkpoint |
| `data` | `generate` | Synthetic data CLI surface — generator bodies deferred |
|        | `inspect` | Dataset inspection — deferred |

Variants live behind `gym.make` kwargs — one gym ID per task:

```python
import gymnasium as gym
import openso101.tasks  # registers gym IDs

env = gym.make("OpenSO101-PickPlace-v0")                          # default RL config
env = gym.make("OpenSO101-PickPlace-v0", cameras=True)            # add wrist + overhead cameras
env = gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")    # 6-DoF joint-position action
env = gym.make("OpenSO101-PickPlace-v0", play=True)               # fewer envs, no domain randomization
env = gym.make("OpenSO101-PickPlace-v0", visual_dr=True)          # lighting + cube-colour DR
```

Visual DR (`randomize_dome_light_intensity`, `randomize_dome_light_color`, `randomize_object_color`) and observation DR (joint-pos / joint-vel noise) are wired on **all three** built-in tasks: Lift, PickPlace, Stack. Physics DR (mass / friction / payload jitter) is attached unconditionally in each task's `__post_init__`.

Register your own task with one decorator:

```python
from openso101.envs import OpenSO101EnvCfg, register_task

@register_task("MyLab-PourTea-v0")
class PourTeaCfg(OpenSO101EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # ... your scene, rewards, terminations ...
```

For deep dives:

**Concepts**
- [Tasks and Environments](docs/concepts/tasks_and_envs.md) — the single-class-per-task pattern and the variant hooks (`configure_play`, `configure_action_mode`, `configure_cameras`, `configure_visual_dr`).
- [RL Algorithms](docs/concepts/rl_algorithms.md) — PPO, why safe-RL stays in `safe_sim2real`.
- [Imitation Learning](docs/concepts/imitation_learning.md) — teleop → HDF5 → LeRobot → ACT/Diffusion via `lerobot.scripts.train`.
- [Sim-to-Real Robustness](docs/concepts/sim2real.md) — DR coverage and the deploy bridge.

**Guides**
- [Installation](docs/guides/install.md) — fresh Ubuntu 22.04 walkthrough; the uv override explanation.
- [Quickstart](docs/guides/quickstart.md) — install to trained PPO checkpoint in 20 min.
- [Teleop setup](docs/guides/teleop.md) — leader-arm wiring, calibration, recording, key bindings.
- [Add a Custom Task](docs/guides/add_a_task.md) — subclass `OpenSO101EnvCfg`, register, configure variants.
- [Migration from `safe_sim2real`](docs/guides/migration_from_safe_sim2real.md) — naming map, CLI map, common gotchas.

**Operating notes**
- [Development diary](docs/development_diary.md) — dated decision log.
- [Isaac Sim learning guide](docs/isaac_sim_learning_guide.md) — Isaac Lab concepts cheat-sheet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

Done:
- [x] Core framework: `OpenSO101EnvCfg`, `register_task`, single-class-per-task pattern
- [x] Built-in tasks: Lift, PickPlace, Stack (all face the cube spawn at startup)
- [x] PPO via rsl_rl with `BestCheckpointRunner` (best-mean-reward snapshotting)
- [x] Teleop boundary: async daemon-polled LeRobot leader arm → simulated follower
- [x] HDF5 + LeRobot dataset recording with checkpoint/restore + interactive save prompt
- [x] IL training via `lerobot.scripts.train` (ACT, Diffusion, any LeRobot policy)
- [x] Same-checkpoint sim playback (`il play`) and real deploy (`sim2real deploy`)
- [x] Visual + observation + physics domain randomization on all three tasks
- [x] uv-based installer that resolves the isaaclab/lerobot packaging conflict cleanly

Deferred (intentionally, until RL + IL are fully validated):
- [ ] Synthetic data generation — MimicGen + Isaac Lab Mimic generator bodies
- [ ] Standalone HDF5 → LeRobot batch converter (inline path runs in `il record` today)

Not planned:
- Off-policy RL (SAC), safe-RL (PPO-Lagrangian/CPO/FOCOPS) — stay in `safe_sim2real`.

See [open issues](https://github.com/KevinYan-831/OpenSO-101/issues) for the live backlog.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. **Any contribution you make is greatly appreciated.**

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag `enhancement`.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Before submitting, please:
- Run `pytest tests/ -v` and make sure all non-skipped tests pass (the suite runs without Isaac Sim).
- Follow the conventions documented in `docs/guides/add_a_task.md`.
- Keep changes scoped — one PR per concern.

### Top contributors

<a href="https://github.com/KevinYan-831/OpenSO-101/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=KevinYan-831/OpenSO-101" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for the full text.

Portions of the code (`src/openso101/robots/so101/`) derive from Isaac Lab's BSD-3-Clause licensed assets; see [`LICENSE-BSD-3-CLAUSE`](LICENSE-BSD-3-CLAUSE).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Jixin Yan — [@KevinYan-831](https://github.com/KevinYan-831)

Project link: [https://github.com/KevinYan-831/OpenSO-101](https://github.com/KevinYan-831/OpenSO-101)

For research collaborations or the legacy safe-RL extension, see the predecessor repository [`safe_sim2real`][safe-sim2real-url].

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

OpenSO-101 stands on the shoulders of a community of open-source projects:

- [NVIDIA Isaac Lab][isaaclab-url] — the simulation and `ManagerBasedRLEnvCfg` substrate.
- [TheRobotStudio SO-ARM100/SO-ARM101][so101-hardware-url] — the open-hardware arm we target.
- [LeRobot][lerobot-url] — teleop drivers, dataset format, ACT/Diffusion training.
- [rsl_rl][rsl-rl-url] — the lean RL library that powers our PPO trainer.
- [`uv`](https://github.com/astral-sh/uv) — the Astral package installer whose `override-dependencies` escape hatch is the only honest way to resolve the isaaclab/lerobot conflict in one pass.
- [MimicGen][mimicgen-url] and [Isaac Lab Mimic][isaaclab-mimic-url] — synthetic data generation (CLI surface wired, bodies deferred).
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template) — the structure of this README.
- The predecessor [`safe_sim2real`][safe-sim2real-url] repository, where most of the implementations were first prototyped.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/KevinYan-831/OpenSO-101.svg?style=for-the-badge
[contributors-url]: https://github.com/KevinYan-831/OpenSO-101/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/KevinYan-831/OpenSO-101.svg?style=for-the-badge
[forks-url]: https://github.com/KevinYan-831/OpenSO-101/network/members
[stars-shield]: https://img.shields.io/github/stars/KevinYan-831/OpenSO-101.svg?style=for-the-badge
[stars-url]: https://github.com/KevinYan-831/OpenSO-101/stargazers
[issues-shield]: https://img.shields.io/github/issues/KevinYan-831/OpenSO-101.svg?style=for-the-badge
[issues-url]: https://github.com/KevinYan-831/OpenSO-101/issues
[license-shield]: https://img.shields.io/github/license/KevinYan-831/OpenSO-101.svg?style=for-the-badge
[license-url]: https://github.com/KevinYan-831/OpenSO-101/blob/main/LICENSE

[python-shield]: https://img.shields.io/badge/python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[isaaclab-shield]: https://img.shields.io/badge/Isaac%20Lab-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[isaaclab-url]: https://github.com/isaac-sim/IsaacLab
[lerobot-shield]: https://img.shields.io/badge/LeRobot-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black
[lerobot-url]: https://github.com/huggingface/lerobot
[rsl-rl-shield]: https://img.shields.io/badge/rsl__rl-2C3E50?style=for-the-badge
[rsl-rl-url]: https://github.com/leggedrobotics/rsl_rl
[gymnasium-shield]: https://img.shields.io/badge/Gymnasium-0078D4?style=for-the-badge
[gymnasium-url]: https://gymnasium.farama.org/

[so101-url]: https://github.com/huggingface/lerobot/blob/main/docs/source/so101.mdx
[so101-hardware-url]: https://github.com/TheRobotStudio/SO-ARM100
[mimicgen-url]: https://github.com/NVlabs/mimicgen
[isaaclab-mimic-url]: https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_mimic
[safe-sim2real-url]: https://github.com/KevinYan-831/safe_sim2real
