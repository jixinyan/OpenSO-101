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
"$task_python" -u -m openso101.cli.main scenes prepare \
    outputs/rl_progress/v3_indexed_scene_exact/bundle \
    --output "outputs/rl_progress/${prefix}_scene" \
    > "outputs/rl_progress/${prefix}_scene.log" 2>&1
test -f "outputs/rl_progress/${prefix}_scene/preparation.json"
"$task_python" -u scripts/check_grasp_task.py \
    --output "outputs/rl_progress/${prefix}_task" \
    --planner-python /home/jixin/workspace/envs/openso101-mujoco/bin/python \
    --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
    > "outputs/rl_progress/${prefix}_task.log" 2>&1
test -f "outputs/rl_progress/${prefix}_task/report.json"
