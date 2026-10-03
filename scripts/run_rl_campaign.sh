set -euo pipefail
cd "${OPENSO101_REPO:?请指定固定提交的代码目录}"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
task_python=/home/jixin/workspace/envs/openso101-v2/bin/python
campaign_name=${1:?请指定训练输出名称}
shift
"$task_python" -u -m openso101.cli.main rl campaign \
    --train-config configs/rl/grasp_v3.json --output "outputs/rl_progress/$campaign_name" \
    --num-envs 2048 --seeds 42 43 44 --gpus "$@"
