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
prefix=${2:?请指定输出名称}
"$task_python" -u -m openso101.cli.main rl distill \
    --teacher-run outputs/rl_progress/v3_checked_probe_8965174 \
    --output "outputs/rl_progress/$prefix" \
    --num-envs 4 --iterations 2 --rollout-steps 16 --headless \
    > "outputs/rl_progress/${prefix}_training.log" 2>&1
test -f "outputs/rl_progress/$prefix/student.json"
"$task_python" -u -m openso101.cli.main rl student-eval \
    --student "outputs/rl_progress/$prefix" --num-envs 4 --n-episodes 8 --headless \
    > "outputs/rl_progress/${prefix}_evaluation.log" 2>&1
test -n "$(find "outputs/rl_progress/$prefix" -maxdepth 1 -name 'evaluation-*.json' -print)"
