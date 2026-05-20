#!/usr/bin/env bash
# One-time environment bootstrap for IL training on Northwestern Quest.
# Run this ONCE on a Quest LOGIN node (compute nodes lack internet for pip).
#
#   ssh <netid>@quest.it.northwestern.edu
#   cd /projects/<ALLOCATION_ID>/OpenSO-101
#   bash scripts/quest_setup.sh
#
# What this does:
#   1. Loads the mamba module (Quest's only supported Python distribution).
#   2. Creates a fresh `openso101` env with Python 3.11.
#   3. Installs torch + lerobot + the openso101 package (editable, no Isaac).
#   4. Strips the broken `groot` policy from lerobot so its policy factory
#      can be imported (upstream LeRobot 0.4.0 ships an unimportable
#      groot_n1.py dataclass — see scripts/quest_setup.sh:patch_groot).
#
# Re-running is safe: it tears down + rebuilds the env. Tunable via env vars:
#   CONDA_ENV_NAME (default: openso101)
#   PYTHON_VERSION (default: 3.11)

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-openso101}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# --------------------------------------------------------------------------
# Sanity: must run on a login node (no GPU here, just network + filesystem).
# --------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    echo "[WARN]: nvidia-smi sees a GPU. quest_setup.sh is meant for LOGIN nodes"
    echo "        (compute nodes lack outbound internet for pip install). If you"
    echo "        meant to be on a login node, this is a no-op warning."
fi

# --------------------------------------------------------------------------
# Load mamba and (re)create the env.
# --------------------------------------------------------------------------
module purge
module load mamba/24.3.0

# Quest's mamba module ships a system-wide read-only package cache at
# /hpc/software/mamba/.../pkgs/cache/cache.lock — libmamba prints noisy
# "Could not open lockfile" errors when it tries to update that cache.
# Point the cache at a user-writable location to silence the warnings.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${HOME}/.conda/pkgs}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-${HOME}/.mamba/pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}" "${MAMBA_PKGS_DIRS}"

if mamba env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "[INFO]: Removing existing env '${CONDA_ENV_NAME}' for a clean rebuild."
    mamba env remove -n "${CONDA_ENV_NAME}" -y
fi

echo "[INFO]: Creating mamba env '${CONDA_ENV_NAME}' (python=${PYTHON_VERSION})"
mamba create -n "${CONDA_ENV_NAME}" -y "python=${PYTHON_VERSION}" pip

# `mamba activate` requires `mamba init` to have been run; `source activate`
# works without it inside bash scripts as long as `module load mamba` is in scope.
source activate "${CONDA_ENV_NAME}"

# --------------------------------------------------------------------------
# Install deps. Order matters: torch first (resolves CUDA build), then
# lerobot (depends on torch), then openso101 (editable; depends on both).
# --------------------------------------------------------------------------
echo "[INFO]: Installing PyTorch (CUDA 12.4 wheels, H100 sm_90 compatible)"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[INFO]: Installing LeRobot 0.4.0 + dataset deps"
# Pin to a known-good LeRobot release to match the local dev environment.
pip install "lerobot==0.4.0" h5py wandb tensorboard

echo "[INFO]: Installing openso101 (editable, IL extras only — Isaac Lab is NOT installed)"
# `pip install -e .` re-uses the pyproject.toml at the repo root. The Isaac
# Lab side of openso101 will import-fail at the gym.register stage if Isaac
# isn't present, but the IL paths (`openso101 il train`) don't touch Isaac.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
pip install -e "${REPO_ROOT}" --no-deps
# Pull in only the runtime deps openso101.il and openso101.cli touch.
pip install gymnasium numpy

# --------------------------------------------------------------------------
# Patch out the broken `groot` policy from lerobot. Same surgery we applied
# on the local install — upstream groot_n1.py has a default-after-non-default
# dataclass field that fails to import, breaking the policy factory for
# every other policy too (eagerly imported by lerobot/policies/__init__.py).
# We don't use GR00T on SO-101, so just remove it.
# --------------------------------------------------------------------------
python - <<'PY'
import sys, shutil
from pathlib import Path

site = Path(__import__("lerobot").__file__).parent
init_py = site / "policies" / "__init__.py"
factory_py = site / "policies" / "factory.py"
groot_dir = site / "policies" / "groot"

def strip_lines(path, predicate):
    text = path.read_text().splitlines(keepends=True)
    new = [l for l in text if not predicate(l)]
    path.write_text("".join(new))

# 1. Remove the groot import line from policies/__init__.py
init_text = init_py.read_text()
init_text = init_text.replace(
    "from .groot.configuration_groot import GrootConfig as GrootConfig\n",
    "# GR00T policy import removed by quest_setup.sh (broken upstream dataclass, unused on SO-101).\n",
)
init_text = init_text.replace('    "GrootConfig",\n', "")
init_py.write_text(init_text)

# 2. Remove top-level + branch references in factory.py
fac = factory_py.read_text()
fac = fac.replace(
    "from lerobot.policies.groot.configuration_groot import GrootConfig\n",
    "# GR00T import removed by quest_setup.sh (broken upstream dataclass).\n",
)
# Remove the `elif name == "groot":` branch in make_policy (a 3-line block).
fac = fac.replace(
    '    elif name == "groot":\n'
    '        from lerobot.policies.groot.modeling_groot import GrootPolicy\n'
    '\n'
    '        return GrootPolicy\n',
    "    # GR00T branch removed by quest_setup.sh.\n",
)
# Remove the `elif policy_type == "groot":` branch in make_policy_config.
fac = fac.replace(
    '    elif policy_type == "groot":\n'
    '        return GrootConfig(**kwargs)\n',
    "    # GR00T branch removed by quest_setup.sh.\n",
)
# Remove the GR00T isinstance check that uses GrootConfig (~20 lines).
# This block is identifiable by its sentinel comment.
import re
fac = re.sub(
    r"\n\s*# TODO\(Steven\):.*?postprocessor_overrides\n\s*return \(",
    "\n        return (",
    fac,
    flags=re.DOTALL,
    count=1,
)
factory_py.write_text(fac)

# 3. rm -rf the groot subpackage.
if groot_dir.exists():
    shutil.rmtree(groot_dir)
    print(f"[INFO]: Removed {groot_dir}")

print("[INFO]: groot patch applied; verifying...")
from lerobot.policies.factory import get_policy_class
for name in ("act", "diffusion"):
    cls = get_policy_class(name)
    print(f"  {name}: {cls.__module__}.{cls.__name__}")
print("[OK] lerobot.policies.factory imports cleanly without groot.")
PY

# --------------------------------------------------------------------------
# Smoke check: can we import everything the SLURM job will need?
# --------------------------------------------------------------------------
echo
echo "[INFO]: Smoke-importing the training entry point..."
python -c "
import torch
import lerobot
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.lerobot_train import train  # entry point used by openso101 il train
print(f'torch={torch.__version__} cuda_compiled={torch.cuda.is_available()}')
print(f'lerobot={lerobot.__version__}')
print(f'ACT class: {get_policy_class(\"act\").__name__}')
print(f'Diffusion class: {get_policy_class(\"diffusion\").__name__}')
print('[OK] All imports succeeded.')
"

echo
echo "[OK] Environment '${CONDA_ENV_NAME}' ready."
echo "    Submit training with: sbatch scripts/quest_train_il.slurm"
