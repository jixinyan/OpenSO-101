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
run=${2:?请指定训练目录}
checkpoint=${3:?请指定已保存模型文件名}
output="outputs/rl_progress/${4:?请指定新的评估目录}"
"$task_python" -u -m openso101.cli.main rl snapshot --run "$run" --checkpoint "$checkpoint" --output "$output"
"$task_python" -u -m openso101.cli.main rl eval \
    --task "${OPENSO101_TASK:-OpenSO101-Lift-v0}" --checkpoint "$output" \
    --num-envs 32 --n-episodes 100 --seed "${5:-10042}" --headless
