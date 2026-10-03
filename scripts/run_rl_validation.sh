set -euo pipefail
cd "${OPENSO101_REPO:-/home/jixin/workspace/code/OpenSO-101-v2}"
export CUDA_VISIBLE_DEVICES="${1:-4}"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
task_python=/home/jixin/workspace/envs/openso101-v2/bin/python
runtime_prefix=${2:-v3_progress}
for runtime_mode in nominal randomized; do
    for runtime_task in Lift PickPlace; do
        runtime_output="outputs/rl_progress/${runtime_prefix}_${runtime_task}_${runtime_mode}"
        "$task_python" -u scripts/check_rl_runtime.py --task "OpenSO101-${runtime_task}-v0" --environment-mode "$runtime_mode" --output "$runtime_output" --steps 500
        test -s "$runtime_output/report.json"
    done
done
