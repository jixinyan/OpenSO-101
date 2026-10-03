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
task_python=/home/jixin/workspace/envs/openso101-v2/bin/python
output="outputs/rl_progress/${2:?请指定输出名称}"
"$task_python" -u -m openso101.cli.main rl train \
    --task OpenSO101-Lift-v0 --backend rsl_rl --algo ppo \
    --train-config configs/rl/grasp_v3.json --task-profile "${5:-grasp_v3}" \
    --seed 42 --num_envs "${4:-32}" --max_iterations "${3:-2}" \
    --output "$output" --headless --no-video --logger tensorboard
test -f "$output/checkpoint.json"
