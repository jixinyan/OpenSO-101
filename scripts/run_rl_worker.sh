set -euo pipefail
cd "${OPENSO101_REPO:?请指定固定提交的代码目录}"
export CUDA_VISIBLE_DEVICES=${1:?请指定 GPU}
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
output=${2:?请指定新的训练输出名称}
seed=${3:?请指定 seed}
iterations=${4:?请指定本次更新数量}
initial_arguments=()
if [[ -n "${5:-}" ]]; then
    initial_arguments=(--load_run "$5")
fi
/home/jixin/workspace/envs/openso101-v2/bin/python -u -m openso101.cli.main rl train \
    --task "${OPENSO101_TASK:-OpenSO101-Lift-v0}" --backend rsl_rl --algo ppo \
    --task-profile grasp_v4 --train-config "${OPENSO101_TRAIN_CONFIG:-configs/rl/grasp_v3.json}" \
    --seed "$seed" --max_iterations "$iterations" --num_envs "${OPENSO101_NUM_ENVS:-2048}" \
    --output "outputs/rl_progress/$output" --headless --no-video --logger tensorboard \
    "${initial_arguments[@]}"
