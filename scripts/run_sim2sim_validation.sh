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
teacher=${2:?请指定已有 teacher 目录}
output="outputs/rl_progress/${3:?请指定新的输出名称}"
environments=${4:-100}
/home/jixin/workspace/envs/openso101-v2/bin/python -u -m openso101.cli.main rl export \
    --task OpenSO101-Lift-v0 --checkpoint "$teacher" --output "${output}_portable" \
    --num-envs "$environments" --validation-steps 300 --seed 30043 --headless \
    > "${output}_export.log" 2>&1
/home/jixin/workspace/envs/openso101-mujoco/bin/python -u -m openso101.cli.main sim2sim mujoco \
    --policy "${output}_portable" --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
    --collision-bundle outputs/rl_progress/gripper_collision --episodes "$environments" \
    --recorded-physics --constrained-drive --output "${output}_mujoco" \
    > "${output}_mujoco.log" 2>&1
