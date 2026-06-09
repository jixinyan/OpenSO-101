#!/usr/bin/env bash
# Fetch the canonical SO-ARM101 USD asset into assets/so101/usd/.
#
# The USD mesh (assets/so101/usd/SO-ARM101-USD.usd, ~23 MB) is a
# third-party binary (the LycheeAI SO-ARM101 mesh, (c) 2025 Muammer Bay
# (LycheeAI) and Louis Le Lay; see LICENSE-BSD-3-CLAUSE) and is NOT
# committed to this repository.
# External users fetch it from the project's GitHub Release.
#
# Resolution order:
#   1. OPENSO101_SO101_USD_SRC -- copy from a local file, if set.
#   2. OPENSO101_SO101_USD_URL -- download from that URL, if set.
#   3. DEFAULT -- download from the project's GitHub Release.
#
# Usage:
#   ./scripts/fetch_so101_usd.sh
#
# Override:
#   OPENSO101_SO101_USD_SRC=/path/to/SO-ARM101-USD.usd ./scripts/fetch_so101_usd.sh
#   OPENSO101_SO101_USD_URL=https://.../SO-ARM101-USD.usd ./scripts/fetch_so101_usd.sh

set -euo pipefail

# --- Editable release coordinates -----------------------------------------
# The GitHub Release the default download pulls from. The repo slug matches
# the configured git remote (jixinyan/OpenSO-101); the tag matches the
# pyproject version (v0.1.0). Override via the env vars below if you host
# the asset elsewhere.
USD_RELEASE_REPO="${OPENSO101_USD_RELEASE_REPO:-jixinyan/OpenSO-101}"
USD_RELEASE_TAG="${OPENSO101_USD_RELEASE_TAG:-v0.1.0}"
USD_RELEASE_URL="https://github.com/${USD_RELEASE_REPO}/releases/download/${USD_RELEASE_TAG}/SO-ARM101-USD.usd"
# --------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST_DIR="${REPO_ROOT}/assets/so101/usd"
DST="${DST_DIR}/SO-ARM101-USD.usd"

mkdir -p "${DST_DIR}"

if [[ -f "${DST}" ]]; then
    echo "[fetch_so101_usd] already present: ${DST}"
    exit 0
fi

download() {
    # download <url>
    local url="$1"
    echo "[fetch_so101_usd] downloading from ${url}"
    if ! curl -fSL -o "${DST}" "${url}"; then
        rm -f "${DST}"
        cat >&2 <<EOF
[fetch_so101_usd] ERROR: download failed from
  ${url}

Check your network connection, or set one of:
  OPENSO101_SO101_USD_SRC=/path/to/SO-ARM101-USD.usd      # local file copy
  OPENSO101_SO101_USD_URL=https://.../SO-ARM101-USD.usd   # alternative URL

The asset is the LycheeAI SO-ARM101 mesh ((c) 2025 Muammer Bay (LycheeAI)
and Louis Le Lay); see LICENSE-BSD-3-CLAUSE for the third-party license
bundled with this repo.
EOF
        exit 1
    fi
}

if [[ -n "${OPENSO101_SO101_USD_SRC:-}" ]]; then
    # (1) Local file copy.
    echo "[fetch_so101_usd] copying from \$OPENSO101_SO101_USD_SRC=${OPENSO101_SO101_USD_SRC}"
    cp "${OPENSO101_SO101_USD_SRC}" "${DST}"
elif [[ -n "${OPENSO101_SO101_USD_URL:-}" ]]; then
    # (2) Explicit remote URL.
    download "${OPENSO101_SO101_USD_URL}"
else
    # (3) DEFAULT: the project's GitHub Release.
    download "${USD_RELEASE_URL}"
fi

echo "[fetch_so101_usd] OK -> ${DST}"
