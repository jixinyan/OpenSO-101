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
prefix=${2:?请指定新的输出名称}
check_kind=${3:?请指定 task 或 physics}
if [[ "$check_kind" == task ]]; then
    camera_arguments=()
    if [[ "${OPENSO101_CAPTURE_CAMERAS:-0}" == 1 ]]; then
        camera_arguments=(--with-cameras --camera-resolution 256)
    fi
    "$task_python" -u scripts/check_grasp_task.py --task-profile grasp_v4 \
        --output "outputs/rl_progress/${prefix}_task" \
        --num-envs "${4:-4}" \
        --planner-python /home/jixin/workspace/envs/openso101-mujoco/bin/python \
        --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml \
        "${camera_arguments[@]}" \
        > "outputs/rl_progress/${prefix}_task.log" 2>&1
elif [[ "$check_kind" == physics ]]; then
    for task in OpenSO101-Lift-v0 OpenSO101-PickPlace-v0; do
        for mode in nominal randomized; do
            "$task_python" -u scripts/check_rl_runtime.py --task-profile grasp_v4 \
                --task "$task" --environment-mode "$mode" --steps 500 \
                --output "outputs/rl_progress/${prefix}_${task}_${mode}" \
                > "outputs/rl_progress/${prefix}_${task}_${mode}.log" 2>&1
        done
    done
else
    exit 1
fi
