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
student=${2:?请指定已完成的 student 目录}
prefix=${3:?请指定新的输出名称}
"$task_python" -u -m openso101.cli.main rl student-eval \
    --student "$student" --num-envs 4 --n-episodes 8 --headless \
    --recording-output "outputs/rl_progress/$prefix" \
    > "outputs/rl_progress/${prefix}.log" 2>&1
"$task_python" -u -m openso101.cli.main sim2real validate \
    --policy-path "$student" --episode "outputs/rl_progress/$prefix/episodes/episode_000000.hdf5" \
    --output "outputs/rl_progress/${prefix}_validation" \
    > "outputs/rl_progress/${prefix}_validation.log" 2>&1
