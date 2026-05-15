# Installation

This guide walks through installing OpenSO-101 from scratch on a fresh
Ubuntu 22.04 workstation. If you already have an Isaac Lab environment, jump
straight to [Step 4: install OpenSO-101](#step-4-install-openso-101).

## Requirements

| Component | Tested version | Why |
|---|---|---|
| OS | Ubuntu 22.04 LTS | Isaac Sim 4.5 / 5.1 official target |
| NVIDIA driver | ≥ 560 | Required by Isaac Sim 5.1 |
| GPU | RTX 20-series or newer, 6 GB+ VRAM | Real-time rendering |
| Python | 3.11 (exact) | Isaac Lab pinned to 3.11 |
| Disk | ~30 GB free | Isaac Sim + Lab + assets |
| Conda / Mambaforge | latest | Isolate Isaac Sim's dep graph |

## Step 1: System Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential git curl libgl1 libglib2.0-0
```

Verify your NVIDIA driver:

```bash
nvidia-smi
# Driver Version: 560.xx or newer, CUDA Version: 12.x
```

## Step 2: Conda / Mambaforge

If you don't already have it:

```bash
curl -fsSLo Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda init bash    # or zsh
```

## Step 3: Isaac Sim + Isaac Lab

The shortest route is the official Isaac Lab pip install, which pulls
Isaac Sim 5.1 as a Python package:

```bash
conda create -n openso101 python=3.11 -y
conda activate openso101

# Isaac Sim (pip-installable since 4.5)
pip install isaacsim==5.1.* --extra-index-url https://pypi.nvidia.com

# Isaac Lab
pip install isaaclab==2.3.* isaaclab_tasks==2.3.* isaaclab_rl==2.3.*
```

Verify the install:

```bash
python -c "from isaaclab.app import AppLauncher; l=AppLauncher(headless=True); l.app.close(); print('isaaclab OK')"
```

For the alternative source-build flow, see Isaac Lab's [official docs][isaaclab-install].

## Step 4: Install OpenSO-101

```bash
git clone https://github.com/KevinYan-831/OpenSO-101.git
cd OpenSO-101
pip install -e .
```

`-e` installs in **editable mode** so changes to `src/openso101/*.py`
take effect without re-installing.

The install registers a `openso101` console script in the env's `bin/`.

## Step 5: External Dependencies

```bash
# rsl_rl for PPO
pip install rsl_rl-lib==2.5.*

# LeRobot for teleop / dataset / IL
pip install lerobot==0.4.*

# Optional: wandb for training logs
pip install wandb
wandb login
```

## Step 6: SO-101 USD Asset

OpenSO-101 expects the canonical SO-ARM101 USD model at
`outputs/third_party/so101_usd/SO-ARM101-USD.usd` (relative to the repo
root). The asset is **not committed** (it's a third-party binary); follow
your team's internal distribution channel or build it from the URDF in
`assets/so101/urdf/` using `omni.isaac.urdf_importer`.

## Step 7: Verify

```bash
openso101 envs list
# Expected output:
# OpenSO101-Lift-v0
# OpenSO101-PickPlace-v0
# OpenSO101-Stack-v0

openso101 rl train --help
openso101 il record --help
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'isaaclab'`** — your shell's
active env is not `openso101`. Run `conda activate openso101`.

**Isaac Sim startup prints `IOMMU is enabled` warnings** — harmless on
most setups. If you see a hang past 30 seconds, try `--headless` or
check your NVIDIA driver version.

**`openso101 envs list` prints nothing** — Isaac Sim crashed silently
during task import. Re-run with stderr visible:
`openso101 envs list 2>&1 | tee /tmp/list.log` and inspect the log.

**Teleop can't open `/dev/ttyACM0`** — add your user to `dialout`:
`sudo usermod -aG dialout $USER` and log out / back in.

[isaaclab-install]: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
