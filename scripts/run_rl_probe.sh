set -euo pipefail
cd "${OPENSO101_REPO:-/home/jixin/workspace/code/OpenSO-101-v2}"
export CUDA_VISIBLE_DEVICES="${1:-4}"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
task_python=/home/jixin/workspace/envs/openso101-v2/bin/python
probe_output="outputs/rl_progress/${2:-v3_training_probe}"
"$task_python" -u -m openso101.cli.main rl train --task OpenSO101-Lift-v0 --algo ppo --backend rsl_rl --train-config configs/rl/grasp_v3.json --task-profile grasp_v3 --num_envs 32 --max_iterations 2 --headless --no-video --logger tensorboard --output "$probe_output"
test -s "$probe_output/checkpoint.json"
test -s "$probe_output/evaluation_history.json"
"$task_python" -u -m openso101.cli.main rl export --task OpenSO101-Lift-v0 --checkpoint "$probe_output" --output "${probe_output}_portable" --validation-steps 300 --headless
test -s "${probe_output}_portable/validation.json"
