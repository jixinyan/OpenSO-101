#!/usr/bin/env bash
# OpenSO-101 install wrapper.
#
# Solves two known pip pathologies in one shot so a fresh-env user can
# run a single command and end up with a fully-resolved environment:
#
#   1. The `isaaclab_rl/setup.py` declares `packaging<24` (an erroneous
#      upper bound — the only `packaging` API both packages use is
#      `packaging.version.Version`, which has been API-stable since
#      v21.x). `lerobot==0.4.0` declares `packaging>=24.2,<26`. These
#      ranges don't intersect, so pip's resolver hits
#      `ResolutionImpossible`. We fix it by using `uv` for the install
#      and declaring `[tool.uv] override-dependencies` in
#      `pyproject.toml` to widen the `packaging` range at solve time.
#      No `--no-deps`, no manually layered dep list — both packages get
#      installed in a single, honest resolver pass.
#
#   2. `setuptools 81+` removed `pkg_resources` from the default
#      install, but some transitive sdists (notably `flatdict`) still
#      `import pkg_resources` in their legacy `setup.py`. We pin
#      `setuptools<81` in pip's isolated build environments via
#      `PIP_CONSTRAINT=$(pwd)/constraints.txt`.
#
# Usage:
#   bash scripts/install.sh           # default: fresh install via uv
#   bash scripts/install.sh --quick   # skip dep resolution; just register openso101
#   bash scripts/install.sh --help    # show this banner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" || ! -f "${REPO_ROOT}/constraints.txt" ]]; then
    echo "[ERROR] $(basename "$0") must run from inside the OpenSO-101 repo." >&2
    echo "        Couldn't find pyproject.toml + constraints.txt under ${REPO_ROOT}." >&2
    exit 1
fi

export PIP_CONSTRAINT="${REPO_ROOT}/constraints.txt"
echo "[install] PIP_CONSTRAINT=${PIP_CONSTRAINT}"

if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[ERROR] No active conda or virtualenv detected." >&2
    echo "        Activate the openso101 env first: conda activate openso101" >&2
    exit 1
fi

mode="${1:-default}"
shift || true

case "${mode}" in
    default|""|full|--full|staged|--staged)
        # `uv` is the modern Python installer (https://github.com/astral-sh/uv);
        # we use it specifically for its `override-dependencies` support
        # which is the only way to resolve the isaaclab/lerobot
        # `packaging` conflict in a single, honest invocation.
        echo "[install] Step 1/3: Bootstrapping uv (used for the isaaclab/lerobot resolver override)..."
        pip install --quiet "uv>=0.5"

        echo "[install] Step 2/3: CUDA-correct torch wheels (cu128, ~3 GB)."
        uv pip install -r "${REPO_ROOT}/requirements-cuda.txt"

        echo "[install] Step 3/3: openso101 + isaaclab + lerobot (single resolver pass)."
        echo "[install]   Expect 15-45 min on first install; Isaac Sim is ~6 GB."
        uv pip install -e "${REPO_ROOT}" "$@"
        ;;
    --quick|quick)
        # User already has a working env (isaaclab + lerobot + torch). Just
        # register our package + console script. Useful for `git pull`
        # workflows where deps haven't changed.
        echo "[install] Quick install (assumes env already has isaaclab + lerobot)."
        pip install -e "${REPO_ROOT}" --no-deps -v "$@"
        ;;
    --help|-h|help)
        cat <<'HELP'
Usage: bash scripts/install.sh [MODE]

Modes:
  default  (full install) Bootstraps uv, then runs a single
                          `uv pip install -e .` that resolves isaaclab +
                          lerobot together using the [tool.uv]
                          override-dependencies block in pyproject.toml.
                          Use this for fresh installs.
  --quick                 Just `pip install -e . --no-deps`. Use when
                          isaaclab and lerobot are already installed and
                          you just need to refresh the editable install
                          (e.g. after `git pull`).

The dep-conflict workaround is documented in pyproject.toml's [tool.uv]
section; see docs/guides/install.md for the why + how.
HELP
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown mode: ${mode}" >&2
        echo "        Run with --help to see available modes." >&2
        exit 2
        ;;
esac

_apply_lerobot_post_install_fixes() {
    # Three fixes that ALL users hit when running `openso101 il train`, applied
    # idempotently so this function is safe to re-run on existing envs.
    #
    # 1. LeRobot 0.4.0 ships an unimportable `groot` policy (its config dataclass
    #    has a default-after-non-default field that fails to construct). Because
    #    `lerobot/policies/__init__.py` eagerly imports every policy class at
    #    module load, that one broken class breaks the factory for ACT,
    #    Diffusion, π0, etc. — every policy you DO want to use. GR00T is a
    #    humanoid foundation model that's irrelevant on SO-101, so we excise it.
    #
    # 2. LeRobot's default video backend is `torchcodec`, which is compiled
    #    against FFmpeg 4-7 and dlopen()s a hard-coded soname (e.g.
    #    libavutil.so.59 for ffmpeg 7). Modern conda-forge ships ffmpeg 8.x
    #    which provides libavutil.so.60 — torchcodec doesn't recognize it and
    #    the dataloader crashes on the first batch. Pin to <8.
    #
    # 3. torchcodec's wheels are built against ONE specific PyTorch minor +
    #    a specific C++ ABI. Compatibility table:
    #      torchcodec 0.4 → torch 2.6 (cu124 wheels, _GLIBCXX_USE_CXX11_ABI=0)
    #      torchcodec 0.5 → torch 2.7 (cu128 wheels, _GLIBCXX_USE_CXX11_ABI=1)
    #      torchcodec 0.6 → torch 2.8
    #    LeRobot 0.4.0 declares `torchcodec>=0.2.1,<0.6.0` so pip picks the
    #    highest matching version. If the torch / torchcodec pair doesn't
    #    match, you get either a missing-soname error (FFmpeg ABI mismatch)
    #    or "undefined symbol: _ZNK3c106Device3strB5cxx11Ev" (C++ ABI
    #    mismatch). The project's requirements-cuda.txt pins torch 2.7+cu128,
    #    so we expect torchcodec 0.5 here — but we detect at runtime so this
    #    function survives future torch upgrades.

    echo "[install] Post-install: patching LeRobot policy factory + FFmpeg."

    python - <<'PY'
import pathlib, shutil, sys

try:
    import lerobot
except ImportError:
    print("[post-install] LeRobot not installed; skipping policy-factory patch.")
    sys.exit(0)

site = pathlib.Path(lerobot.__file__).parent
groot_dir = site / "policies" / "groot"
init_py = site / "policies" / "__init__.py"
factory_py = site / "policies" / "factory.py"

# 1. Drop the eager import of GrootConfig from policies/__init__.py.
if init_py.exists():
    text = init_py.read_text()
    new = text.replace(
        "from .groot.configuration_groot import GrootConfig as GrootConfig\n",
        "# groot import removed by openso101 install.sh (broken upstream dataclass, unused on SO-101).\n",
    ).replace('    "GrootConfig",\n', "")
    if new != text:
        init_py.write_text(new)
        print(f"[post-install] Patched {init_py.name}: removed GrootConfig import.")

# 2. Drop top-level + branch references in policies/factory.py.
if factory_py.exists():
    text = factory_py.read_text()
    new = text.replace(
        "from lerobot.policies.groot.configuration_groot import GrootConfig\n",
        "# groot import removed by openso101 install.sh.\n",
    ).replace(
        '    elif name == "groot":\n'
        '        from lerobot.policies.groot.modeling_groot import GrootPolicy\n'
        '\n'
        '        return GrootPolicy\n',
        "    # groot branch removed by openso101 install.sh.\n",
    ).replace(
        '    elif policy_type == "groot":\n'
        '        return GrootConfig(**kwargs)\n',
        "    # groot branch removed by openso101 install.sh.\n",
    )
    if new != text:
        factory_py.write_text(new)
        print(f"[post-install] Patched {factory_py.name}: removed groot branches.")

# 3. Remove the groot subpackage entirely so leftover imports surface
#    immediately rather than masquerading as missing-attribute errors.
if groot_dir.exists():
    shutil.rmtree(groot_dir)
    print(f"[post-install] Removed {groot_dir}.")

# 4. Verify the factory imports cleanly.
from lerobot.policies.factory import get_policy_class
for name in ("act", "diffusion"):
    get_policy_class(name)
print("[post-install] lerobot.policies.factory imports cleanly (act + diffusion).")
PY

    # Auto-pin torchcodec to the version matching the installed torch.
    # If we can't determine torch, we skip — the user will see the symbol
    # error and at least our error message points them at the cause.
    python - <<'PY'
import importlib, subprocess, sys

try:
    import torch
except ImportError:
    print("[post-install] torch not installed; skipping torchcodec pin.")
    sys.exit(0)

# torch.__version__ looks like "2.6.0+cu124"; split off the local version tag.
torch_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])  # "2.6"
torchcodec_for_torch = {"2.6": "0.4", "2.7": "0.5", "2.8": "0.6"}.get(torch_minor)
if torchcodec_for_torch is None:
    print(f"[post-install] torch {torch_minor} has no known torchcodec mapping; "
          f"skipping pin (test torchcodec import yourself).")
    sys.exit(0)

# Only reinstall if the currently-installed version is wrong.
try:
    tc = importlib.import_module("torchcodec")
    tc_minor = ".".join(tc.__version__.split(".")[:2]) if hasattr(tc, "__version__") else None
except Exception:
    tc_minor = None

if tc_minor == torchcodec_for_torch:
    print(f"[post-install] torchcodec=={tc_minor} already matches torch {torch_minor}.")
    sys.exit(0)

print(f"[post-install] Pinning torchcodec=={torchcodec_for_torch}.* to match torch {torch_minor}.")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                       f"torchcodec=={torchcodec_for_torch}.*"])
PY

    # Pin ffmpeg<8 for torchcodec compatibility. Conda/mamba envs only;
    # non-conda users need to install FFmpeg 7.x system-wide themselves.
    if [[ -n "${CONDA_PREFIX:-}" ]] && command -v mamba >/dev/null 2>&1; then
        echo "[install] Pinning ffmpeg<8 (torchcodec needs FFmpeg 4-7)."
        mamba install -y -n "$(basename "${CONDA_PREFIX}")" -c conda-forge 'ffmpeg<8' || \
            mamba install -y -p "${CONDA_PREFIX}" -c conda-forge 'ffmpeg<8'
    elif [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
        echo "[install] Pinning ffmpeg<8 (torchcodec needs FFmpeg 4-7)."
        conda install -y -p "${CONDA_PREFIX}" -c conda-forge 'ffmpeg<8'
    else
        echo "[install] WARNING: no conda/mamba detected; install FFmpeg 7.x"
        echo "          system-wide yourself before running 'openso101 il train',"
        echo "          or LeRobot's dataloader will crash with"
        echo "          'libavutil.so.59: cannot open shared object file'."
    fi

    # Confirm torchcodec can actually decode video now.
    if python -c "from torchcodec.decoders import VideoDecoder" 2>/dev/null; then
        echo "[install] torchcodec imports cleanly. IL training is ready."
    else
        echo "[install] WARNING: torchcodec import still failing. Check FFmpeg install."
    fi
}

_apply_lerobot_post_install_fixes

echo "[install] Done. Verify with: openso101 envs list"
