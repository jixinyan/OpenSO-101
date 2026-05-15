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
    State-of-the-art open-source robot learning framework for the LeRobot SO-101 in Isaac Lab.
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

1. **Reinforcement Learning** — PPO via [`rsl_rl`][rsl-rl-url] with PPO-friendly defaults for SO-101; SAC slated for sub-project E.
2. **Imitation Learning** — leader-arm teleoperation, HDF5 recording, [LeRobot dataset][lerobot-url] export, ACT / Diffusion training (sub-project C).
3. **Synthetic Data Generation** — [MimicGen][mimicgen-url] and [Isaac Lab Mimic][isaaclab-mimic-url] adapters that turn a handful of human demos into thousands of trajectories (sub-project F).
4. **Sim-to-Real Robustness** — phase-1 domain randomization built in; deeper transfer tooling lands in sub-project B.

The project is organized so that a researcher can clone, install, and reach a working `openso101 envs list` in five minutes — and so a downstream contributor can register a custom task with one decorator. Each pillar exposes a stable CLI verb and a stable Python entry point; swapping in a custom algorithm or task does not require forking the framework.

Out of scope (lives in the legacy [`safe_sim2real`][safe-sim2real-url] repository):
- Safe-RL (PPO-Lagrangian, CPO, FOCOPS).
- SB3 / StableBaselines3 wrappers.

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
│   ├── robots/so101/         # SO-101 ArticulationCfg, USD spawn helpers, cameras
│   ├── tasks/                # Built-in tasks: lift, pick_place, stack (+ shared/)
│   ├── teleop/               # LeRobot leader-arm → simulated follower
│   ├── rl/                   # rsl_rl-backed PPO + runners (BestCheckpointRunner)
│   ├── il/                   # ACT / Diffusion policies, LeRobot dataset adapter
│   ├── data_gen/             # MimicGen + Isaac Lab Mimic adapters
│   └── sim2real/             # Domain randomization, deployment placeholders
├── docs/                     # Concepts, guides, dev diary, this README's siblings
├── examples/                 # End-to-end recipe scripts
├── tests/                    # pytest suite
└── pyproject.toml            # `openso101` console_scripts entry point
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- **Hardware:** an NVIDIA GPU (RTX 20-series or newer; CUDA 12.x). 6 GB+ VRAM recommended for play, 16 GB+ for training.
- **OS:** Ubuntu 22.04 LTS (Isaac Sim 4.5 / 5.1 official target).
- **Driver:** NVIDIA driver ≥ 560.
- **Python:** 3.11 (Isaac Lab is pinned to 3.11).
- **Conda or Mambaforge:** recommended to keep Isaac Sim's dependency graph isolated.

### Installation

OpenSO-101 reuses the Isaac Lab installation pattern. The shortest path:

```bash
# 1. Clone
git clone https://github.com/KevinYan-831/OpenSO-101.git
cd OpenSO-101

# 2. Create the env (Isaac Sim, Isaac Lab, rsl_rl, lerobot)
conda env create -f environment.yml
conda activate openso101

# 3. Install OpenSO-101 in editable mode
pip install -e .

# 4. Verify
openso101 envs list
# OpenSO101-Lift-v0
# OpenSO101-PickPlace-v0
# OpenSO101-Stack-v0
```

If you already have an Isaac Lab environment, you can install OpenSO-101 on top of it with `pip install -e .` directly — the `environment.yml` is just a convenience.

### Quickstart

**Train PPO on PickPlace** (headless, single GPU):

```bash
openso101 rl train --task OpenSO101-PickPlace-v0 --algo ppo --headless
```

**Replay the best checkpoint:**

```bash
openso101 rl play --task OpenSO101-PickPlace-v0 \
  --checkpoint logs/rsl_rl/openso101_pickplace/2026-05-14_12-30-00/model_best.pt
```

**Record a teleop demonstration** with a real SO-101 leader arm:

```bash
openso101 il record \
  --task OpenSO101-PickPlace-v0 \
  --leader-port /dev/ttyACM0 \
  --leader-id leader_arm_1 \
  --repo-root teleop_data/openso101_pickplace
```

**Push the dataset to the Hugging Face Hub:**

```bash
openso101 il push --repo-root teleop_data/openso101_pickplace --repo-id <username>/openso101-pickplace
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
| `rl` | `train` | Train an RL policy (`--algo {ppo,sac}`) |
|      | `play` | Replay a trained checkpoint |
|      | `plot` | Plot training curves from a run dir |
| `il` | `record` | Record teleop demonstrations to HDF5 + LeRobot |
|      | `push` | Push dataset to the Hugging Face Hub |
|      | `train` | Train an IL policy (ACT, Diffusion) — sub-project C |
|      | `play` | Replay a trained IL policy in sim |
|      | `replay` | Replay a recorded teleop episode |
| `data` | `generate` | Synthetic data generation — sub-project F |
| `sim2real` | `deploy` | Deploy a policy to the real SO-101 — future |

Variants live behind `gym.make` kwargs — one gym ID per task:

```python
import gymnasium as gym
import openso101.tasks  # registers gym IDs

env = gym.make("OpenSO101-PickPlace-v0")                     # default RL config
env = gym.make("OpenSO101-PickPlace-v0", cameras=True)       # add wrist + overhead cameras
env = gym.make("OpenSO101-PickPlace-v0", action_mode="teleop")  # 6-DoF joint-position action
env = gym.make("OpenSO101-PickPlace-v0", play=True)          # fewer envs, no domain randomization
```

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
- [Tasks and Environments](docs/concepts/tasks_and_envs.md) — the single-class-per-task pattern and the three variant hooks.
- [RL Algorithms](docs/concepts/rl_algorithms.md) — PPO, SAC roadmap, why safe-RL stays in `safe_sim2real`.
- [Imitation Learning](docs/concepts/imitation_learning.md) — teleop → HDF5 → LeRobot → ACT/Diffusion.
- [Synthetic Data Generation](docs/concepts/data_generation.md) — MimicGen and Isaac Lab Mimic adapters.
- [Sim-to-Real Robustness](docs/concepts/sim2real.md) — phase-1 DR today, phase-2 deploy bridge later.

**Guides**
- [Installation](docs/guides/install.md) — fresh Ubuntu 22.04 walkthrough.
- [Quickstart](docs/guides/quickstart.md) — install to trained PPO checkpoint in 20 min.
- [Teleop setup](docs/guides/teleop.md) — leader-arm wiring, calibration, recording.
- [Add a Custom Task](docs/guides/add_a_task.md) — subclass `OpenSO101EnvCfg`, register, configure variants.
- [Migration from `safe_sim2real`](docs/guides/migration_from_safe_sim2real.md) — naming map, CLI map, common gotchas.

**Operating notes**
- [Development diary](docs/development_diary.md) — dated decision log.
- [Isaac Sim learning guide](docs/isaac_sim_learning_guide.md) — Isaac Lab concepts cheat-sheet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] Core framework: `OpenSO101EnvCfg`, `register_task`, single-class-per-task pattern
- [x] Built-in tasks: Lift, PickPlace, Stack
- [x] PPO via rsl_rl with `BestCheckpointRunner`
- [x] Teleop boundary: LeRobot leader arm → simulated follower
- [x] HDF5 + LeRobot dataset recording
- [ ] Sub-project B — sim-to-real transfer (action/observation noise, ROS bridge)
- [ ] Sub-project C — IL training (ACT, Diffusion)
- [ ] Sub-project E — off-policy RL (SAC)
- [ ] Sub-project F — synthetic data generation (MimicGen, Isaac Lab Mimic)

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
- Run `pytest tests/ -v` and make sure all non-skipped tests pass.
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
- [LeRobot][lerobot-url] — teleop drivers, dataset format, ACT/Diffusion training references.
- [rsl_rl][rsl-rl-url] — the lean RL library that powers our PPO trainer.
- [MimicGen][mimicgen-url] and [Isaac Lab Mimic][isaaclab-mimic-url] — synthetic data generation.
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
