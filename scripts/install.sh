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

echo "[install] Done. Verify with: openso101 envs list"
