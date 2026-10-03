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
teacher=${2:?请指定经过文件校验的 teacher 目录}
prefix=${3:?请指定新的输出名称}
"$task_python" -u -m openso101.cli.main rl validate-loop \
    --teacher-run "$teacher" --output "outputs/rl_progress/$prefix" \
    --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
    --collision-bundle outputs/rl_progress/gripper_collision \
    --mujoco-python /home/jixin/workspace/envs/openso101-mujoco/bin/python
