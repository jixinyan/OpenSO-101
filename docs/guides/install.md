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

## Step 3: Create the conda env

```bash
git clone https://github.com/KevinYan-831/OpenSO-101.git
cd OpenSO-101

conda env create -f environment.yml
conda activate openso101
```

The `environment.yml` here only pins Python 3.11; the heavy AI/sim stack
ships best as pip wheels and is installed in the next steps.

## Step 4: Install OpenSO-101

```bash
# 1. CUDA-correct PyTorch (must come BEFORE `pip install -e .` so the
#    cu128 wheel wins over PyPI's default cu126 wheel — matters on
#    Blackwell GPUs / RTX 50-series; safe on every other GPU too).
pip install -r requirements-cuda.txt

# 2. Install OpenSO-101 in editable mode. This pulls Isaac Sim, Isaac
#    Lab, rsl_rl, lerobot, etc. from the dependency list in pyproject.toml.
pip install -e .
```

`-e` installs in **editable mode** so changes to `src/openso101/*.py`
take effect without re-installing. The install registers a `openso101`
console script in the env's `bin/`.

Verify Isaac Lab is reachable:

```bash
python -c "from isaaclab.app import AppLauncher; l=AppLauncher(headless=True); l.app.close(); print('isaaclab OK')"
```

For the alternative Isaac Lab source-build flow, see [Isaac Lab's official docs][isaaclab-install].

## Step 5: Optional Logging Tools

```bash
# wandb for training dashboards (otherwise tensorboard is used)
pip install wandb
wandb login
```

## Step 6: SO-101 USD Asset

OpenSO-101 expects the canonical SO-ARM101 USD model at
`assets/so101/usd/SO-ARM101-USD.usd` (relative to the repo root). The
asset is a 23 MB third-party binary (authored by [Lior Ben Horin][lbh],
MIT-licensed) and is **not committed**.

Three ways to provision it:

```bash
# Option A: you have a sibling safe_sim2real checkout — easiest path.
./scripts/fetch_so101_usd.sh

# Option B: you have the .usd file locally somewhere else.
OPENSO101_SO101_USD_SRC=/path/to/SO-ARM101-USD.usd ./scripts/fetch_so101_usd.sh

# Option C: download from a URL you control.
OPENSO101_SO101_USD_URL=https://example.com/SO-ARM101-USD.usd ./scripts/fetch_so101_usd.sh
```

Alternative: override the lookup path entirely without copying the file:

```bash
export OPENSO101_SO101_USD_PATH=/some/other/location/SO-ARM101-USD.usd
```

If you want to rebuild the USD from the URDF in `assets/so101/urdf/`,
use Isaac Sim's `omni.isaac.urdf_importer` extension — the URDF was
imported with the convex-decomposition collision settings noted in
`docs/development_diary.md`.

[lbh]: https://github.com/lbh-rs

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
