set -euo pipefail
cd "${OPENSO101_REPO:?请指定代码目录}"
export CUDA_VISIBLE_DEVICES=${1:?请指定 GPU}
export OMNI_KIT_ACCEPT_EULA=YES
export OPENSO101_SO101_USD_PATH="$PWD/outputs/SO-ARM101-USD.usd"
export TMPDIR="$PWD/outputs/tmp"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
physical_gpu=$CUDA_VISIBLE_DEVICES
runtime_lib="$PWD/outputs/ffmpeg/root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="/home/jixin/workspace/envs/openso101-v2/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$runtime_lib:$runtime_lib/pulseaudio:$runtime_lib/blas:$runtime_lib/lapack:/home/jixin/workspace/envs/edh-graphics/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
shift
native_python=/home/jixin/workspace/envs/openso101-v2/bin/python
guard_options=()
if [[ -n "${OPENSO101_GPU_LAUNCH_REPORT:-}" ]]; then
    guard_options=(--report "$OPENSO101_GPU_LAUNCH_REPORT")
fi
exec "$native_python" -u scripts/run_idle_gpu.py --gpu "$physical_gpu" "${guard_options[@]}" -- "$native_python" -u "$@"
