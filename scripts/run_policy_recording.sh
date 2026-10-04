set -euo pipefail
cd "${OPENSO101_REPO:?请指定代码目录}"
export CUDA_VISIBLE_DEVICES=${1:?请指定 GPU}
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
task_python=/home/jixin/workspace/envs/openso101-v2/bin/python
checkpoint=${2:?请指定完整 checkpoint 目录}
output=${3:?请指定新的录制目录}
"$task_python" -u -m openso101.cli.main rl eval \
    --task "${OPENSO101_TASK:-OpenSO101-Lift-v0}" --checkpoint "$checkpoint" \
    --num-envs 4 --n-episodes 4 --seed "${4:-10043}" --headless \
    --recording-output "$output" --camera-resolution 512
