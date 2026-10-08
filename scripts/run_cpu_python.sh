set -euo pipefail
cd "${OPENSO101_REPO:?请指定代码目录}"
runtime=${1:?请指定 native 或 mujoco 环境}
case "$runtime" in
    native) cpu_python=/home/jixin/workspace/envs/openso101-v2/bin/python ;;
    mujoco) cpu_python=/home/jixin/workspace/envs/openso101-mujoco/bin/python ;;
    *) exit 2 ;;
esac
shift
export CUDA_VISIBLE_DEVICES=""
export OPENSO101_SKIP_ISAAC=1
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=osmesa
runtime_lib="$PWD/outputs/ffmpeg/root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$PWD/outputs/dependencies/osmesa/root/usr/lib/x86_64-linux-gnu:$runtime_lib:$runtime_lib/pulseaudio:$runtime_lib/blas:$runtime_lib/lapack:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "$cpu_python" -u "$@"
