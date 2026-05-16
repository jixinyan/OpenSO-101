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
| RAM | 32 GB+ recommended | Isaac Sim's per-env scene replication |
| Python | 3.11 (exact) | Isaac Lab pinned to 3.11 |
| Disk | ~30 GB free | Isaac Sim + Lab + assets |
| Conda / Mambaforge | latest | Isolate Isaac Sim's dep graph |

### Memory budget for RL training

The default `num_envs` for built-in tasks is **4096**, sized for workstation
GPUs (24 GB+ VRAM, 64 GB+ RAM). On smaller hardware you must pass
`--num_envs <N>` to fit:

| Configuration | Safe `--num_envs` on 8 GB VRAM / 32 GB RAM | Notes |
|---|---|---|
| `rl train` (state-only, default) | up to 2048 | No rendering pipeline |
| `rl train --visual-dr` | 64 to 128 | RTX pipeline initialized; per-env memory grows ~5x |
| `rl train --with-cameras` | 16 to 32 | Actual camera rendering per env |
| `rl train --visual-dr --with-cameras` | 8 to 16 | Both costs combined |

If you see the process die with `Killed` (no traceback), that's the
Linux OOM killer hitting RAM. Reduce `--num_envs`. If you see a CUDA
`out of memory` error, that's VRAM; same fix.

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

**Heads-up on duration.** This step is the slow one — expect **15–45
minutes** end-to-end on a typical residential connection. The big costs
are the Isaac Sim wheels (~6 GB) and dependency resolution.

Run the install wrapper:

```bash
bash scripts/install.sh
```

That is exactly equivalent to running, in order:

```bash
export PIP_CONSTRAINT=$(pwd)/constraints.txt
pip install uv                                     # bootstrap uv
uv pip install -r requirements-cuda.txt            # torch cu128
uv pip install -e .                                # openso101 + isaaclab + lerobot
```

`uv` is the modern Python installer from Astral. We use it
specifically because pip's resolver hits `ResolutionImpossible` on the
isaaclab + lerobot combination (see "Why `uv`?" below); uv resolves
the conflict in a single, honest invocation via the
`[tool.uv] override-dependencies` block declared in `pyproject.toml`.

If your env already has `isaaclab` and `lerobot` installed and you
just want to refresh the editable install (e.g. after `git pull`):

```bash
bash scripts/install.sh --quick
```

### Why `uv`?

`isaaclab_rl/setup.py` (pulled in by `isaaclab[all,isaacsim]==2.3.0`)
declares `packaging<24`. `lerobot==0.4.0` declares
`packaging>=24.2,<26.0`. These ranges **don't intersect**, so pip's
resolver — which treats every declared bound as authoritative — cannot
derive a combined dependency set and exits with
`ResolutionImpossible`.

The `<24` cap in `isaaclab_rl` is an **erroneous upper bound**: the
only API in `packaging` that either project uses is
`packaging.version.Version`, which has been API-stable since v21.x.
The actual breaking change in `packaging` happened at v22.0
(LegacyVersion removal), not v24.0 — so both projects run fine
against `packaging` 24.x and 25.x. We track upstream relaxation at
[isaac-sim/IsaacLab#5084][isaaclab-pkg-issue].

`uv`'s `override-dependencies` is the documented escape hatch for
exactly this situation: at solve time, uv rewrites the conflicting
transitive metadata to a range we declare in `pyproject.toml`
(`packaging>=24.2,<26`), and the resolver succeeds. No `--no-deps`,
no manually layered fallback dep list — both packages get installed
in a single, honest resolver pass.

`-e` installs OpenSO-101 in **editable mode** so changes to
`src/openso101/*.py` take effect without re-installing. The install
registers an `openso101` console script in the env's `bin/`.

[isaaclab-pkg-issue]: https://github.com/isaac-sim/IsaacLab/issues/5084

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
use Isaac Sim's `omni.isaac.urdf_importer` extension with
convex-decomposition collision settings (mesh approximation: convex
decomposition; max convex hulls: 32; collision filter on the cube).

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

### Install hangs

**`bash scripts/install.sh` appears stuck.** Re-run with verbose
output:

```bash
bash scripts/install.sh -- -v   # `-v` passes through to uv pip install
```

The two normal long phases are:
1. **Resolver run** — uv reads the dep tree and downloads metadata.
   Typically 30 seconds to 2 minutes thanks to uv's parallel fetcher.
2. **Isaac Sim download** — a ~6 GB tarball from `pypi.nvidia.com`.
   Slow connections see 30+ minutes here. The progress bar may stall
   on single chunks but the byte counter should keep moving.

If you must bypass `scripts/install.sh` entirely (e.g. for CI
debugging), the equivalent commands are:

```bash
export PIP_CONSTRAINT=$(pwd)/constraints.txt
pip install "uv>=0.5"
uv pip install -r requirements-cuda.txt
uv pip install -e .
```

Do NOT use plain `pip install -e .` — pip's resolver will hit
`ResolutionImpossible` on the `isaaclab` ↔ `lerobot` packaging
conflict (see "Why `uv`?" in Step 4 above for the root cause).

**`ResolutionImpossible: packaging<24 conflicts with packaging>=24.2`**
— you ran `pip install` instead of `uv pip install`. Use `bash
scripts/install.sh` or the `uv pip install` equivalent above.

**`ModuleNotFoundError: No module named 'pkg_resources'` during
`flatdict` (or similar) wheel build** — setuptools 81 removed
`pkg_resources` from the default install but old sdists still import
it. The fix is `PIP_CONSTRAINT=$(pwd)/constraints.txt` on the install
command; `scripts/install.sh` sets this automatically. If you bypass
the script, set it yourself.

**`Could not find a version that satisfies the requirement
isaacsim-*`** — your pip/uv's index list doesn't include
`pypi.nvidia.com`. uv inherits pip's index config; check
`~/.config/pip/pip.conf` or pass `--extra-index-url
https://pypi.nvidia.com` to `uv pip install`.

**`tensordict` or `torch` wheel mismatches** — almost always a stale
cache. Clear with `uv cache clean` (or `pip cache purge` if pip is
mixed in) and retry. If you've manually installed torch before,
uninstall it first: `uv pip uninstall torch torchvision torchaudio`.

### Runtime issues

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

**`openso101 il record` works but feels laggy** — confirm the async
leader worker is running by adding `--profile-teleop`; the per-second
profile line should report `leader_mode=async polls_this_interval=N`
with N around 100+. If it reports `leader_mode=sync`, the new code
isn't installed — re-run `pip install -e .` to pick up the editable
install changes.

[isaaclab-install]: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
