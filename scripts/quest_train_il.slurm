#!/bin/bash
# SLURM job script for OpenSO-101 IL training on Northwestern Quest.
# Trains ACT then Diffusion back-to-back on a single H100, using a
# Globus-transferred LeRobot dataset under /projects.
#
# Submit from a Quest login node (after running scripts/quest_setup.sh once):
#   sbatch scripts/quest_train_il.slurm
#
# Tunables: set OPENSO101_ROOT and ALLOCATION_ID below to match your paths.
#
#SBATCH --job-name=openso101_il_train
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=sxm
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/openso101_logs/slurm_%j.out
#SBATCH --error=/scratch/%u/openso101_logs/slurm_%j.err
# Uncomment + fill in for email notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=<your-netid>@northwestern.edu
# Uncomment + fill in for project-allocation accounting:
# #SBATCH --account=<ALLOCATION_ID>

set -euo pipefail

# --------------------------------------------------------------------------
# User-configurable paths. Adjust these to match your Quest layout.
# --------------------------------------------------------------------------
# ALLOCATION_ID is your Quest project id (e.g. "p32123"). Used for storage
# paths under /projects/<ALLOCATION_ID>. Replace before submitting.
ALLOCATION_ID="${ALLOCATION_ID:-<ALLOCATION_ID>}"
OPENSO101_ROOT="${OPENSO101_ROOT:-/projects/${ALLOCATION_ID}/OpenSO-101}"
DATASET_ROOT="${DATASET_ROOT:-${OPENSO101_ROOT}/teleop_data/pickplace_v1/lerobot_dataset}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/pickplace_v1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-openso101}"

# Training output goes to /scratch (5 TB, auto-cleans after 30 d). Each
# submission gets its own timestamped subdir so re-runs don't collide.
RUN_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TRAIN_LOG_BASE="/scratch/${USER}/openso101_logs/${RUN_STAMP}"
mkdir -p "${TRAIN_LOG_BASE}"

# --------------------------------------------------------------------------
# Environment setup (Quest convention: mamba, no /home dependency).
# --------------------------------------------------------------------------
module purge
module load mamba/24.3.0
# mamba's `source activate` works inside SLURM jobs without `mamba init`.
source activate "${CONDA_ENV_NAME}"

# Keep HuggingFace cache on /projects so a) compute nodes don't need
# internet to load models/datasets, and b) it survives across jobs.
export HF_HOME="/projects/${ALLOCATION_ID}/.hf-cache"
mkdir -p "${HF_HOME}"

# Unbuffered output so slurm_<jobid>.out is live-tailable via `tail -f`.
export PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------
# Pre-flight checks. Bail loudly before consuming wall time on a broken env.
# --------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] Pre-flight checks"
nvidia-smi
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}')"
python -c "from lerobot.policies.factory import get_policy_class; get_policy_class('act'); get_policy_class('diffusion'); print('lerobot policies OK')" || {
    echo '[ERROR] lerobot.policies.factory import failed.'
    echo '        Did you run scripts/quest_setup.sh (which strips the broken groot policy)?'
    exit 1
}
test -d "${DATASET_ROOT}" || {
    echo "[ERROR] Dataset not found at ${DATASET_ROOT}"
    echo "        Did Globus finish transferring teleop_data/?"
    exit 1
}

cd "${OPENSO101_ROOT}"
echo "[$(date +%H:%M:%S)] Working dir: $(pwd)"
echo "[$(date +%H:%M:%S)] Output base: ${TRAIN_LOG_BASE}"

# --------------------------------------------------------------------------
# Stage 1: train ACT (Action Chunking Transformer).
# --------------------------------------------------------------------------
ACT_OUT="${TRAIN_LOG_BASE}/act_pickplace"
echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] Stage 1: ACT training -> ${ACT_OUT}"
echo "================================================================"

openso101 il train \
    --policy act \
    --dataset "${DATASET_ROOT}" \
    --repo-id "${DATASET_REPO_ID}" \
    --output-dir "${ACT_OUT}" \
    --steps 100000 \
    --batch-size 32 \
    -- \
    --policy.chunk_size=50 --policy.n_action_steps=50

echo "[$(date +%H:%M:%S)] Stage 1 done. ACT checkpoint:"
echo "    ${ACT_OUT}/checkpoints/last/pretrained_model/"

# --------------------------------------------------------------------------
# Stage 2: train Diffusion Policy.
# --------------------------------------------------------------------------
# horizon=48 (NOT 50) because diffusion's UNet has 3 downsampling levels
# (down_dims=(512, 1024, 2048)) and the horizon must be divisible by 2^3=8.
DIFF_OUT="${TRAIN_LOG_BASE}/diffusion_pickplace"
echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] Stage 2: Diffusion training -> ${DIFF_OUT}"
echo "================================================================"

openso101 il train \
    --policy diffusion \
    --dataset "${DATASET_ROOT}" \
    --repo-id "${DATASET_REPO_ID}" \
    --output-dir "${DIFF_OUT}" \
    --steps 100000 \
    --batch-size 16 \
    -- \
    --policy.horizon=48 --policy.n_action_steps=48

echo "[$(date +%H:%M:%S)] Stage 2 done. Diffusion checkpoint:"
echo "    ${DIFF_OUT}/checkpoints/last/pretrained_model/"

echo
echo "================================================================"
echo "[$(date +%H:%M:%S)] All training stages complete."
echo "Outputs under: ${TRAIN_LOG_BASE}"
echo "Pull them back with Globus, then run \`openso101 il play\` locally."
echo "================================================================"
