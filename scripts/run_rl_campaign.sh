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
loop_arguments=()
initial_arguments=()
if [[ -n "${OPENSO101_INITIAL_RUNS:-}" ]]; then
    initial_arguments=(--initial-runs "$OPENSO101_INITIAL_RUNS")
fi
if [[ "${OPENSO101_VALIDATE_LOOP:-0}" == 1 ]]; then
    loop_arguments=(--validate-loop --robot-model outputs/so-arm100/Simulation/SO101/so101_old_calib.xml
                    --collision-bundle outputs/rl_progress/gripper_collision
                    --mujoco-python /home/jixin/workspace/envs/openso101-mujoco/bin/python)
fi
"$task_python" -u -m openso101.cli.main rl campaign \
    --train-config configs/rl/grasp_v3.json --output "outputs/rl_progress/$campaign_name" \
    --task-profile "${OPENSO101_TASK_PROFILE:-grasp_v4}" --num-envs "${OPENSO101_NUM_ENVS:-2048}" \
    --seeds 42 43 44 --gpus "$@" "${loop_arguments[@]}" "${initial_arguments[@]}"
