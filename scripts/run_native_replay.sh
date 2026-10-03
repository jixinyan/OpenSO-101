set -euo pipefail
cd "${OPENSO101_REPO:?请指定固定提交的代码目录}"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
/home/jixin/workspace/envs/openso101-mujoco/bin/python -u scripts/replay_native_grasp.py "$@"
