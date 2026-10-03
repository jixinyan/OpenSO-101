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
prefix=${2:?请指定独立输出名称}
for condition in nominal randomized; do
    for task in PickPlace Lift; do
        output="outputs/rl_progress/${prefix}_${task}_${condition}"
        "$task_python" -u scripts/check_rl_runtime.py \
            --task "OpenSO101-${task}-v0" --environment-mode "$condition" \
            --steps 500 --output "$output" > "${output}.log" 2>&1
        test -f "$output/report.json"
    done
done
