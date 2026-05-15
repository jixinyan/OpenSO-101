#!/usr/bin/env bash
# Fetch the canonical SO-ARM101 USD asset into assets/so101/usd/.
#
# The USD is a third-party binary (~23 MB) authored upstream and is
# intentionally not committed to this repository. This script either
# copies it from a sibling `safe_sim2real` checkout (the most common
# path during the OpenSO-101 transition) or, if you set
# OPENSO101_SO101_USD_URL, downloads from that URL.
#
# Usage:
#   ./scripts/fetch_so101_usd.sh
#
# Override:
#   OPENSO101_SO101_USD_SRC=/path/to/SO-ARM101-USD.usd ./scripts/fetch_so101_usd.sh
#   OPENSO101_SO101_USD_URL=https://.../SO-ARM101-USD.usd ./scripts/fetch_so101_usd.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST_DIR="${REPO_ROOT}/assets/so101/usd"
DST="${DST_DIR}/SO-ARM101-USD.usd"

mkdir -p "${DST_DIR}"

if [[ -f "${DST}" ]]; then
    echo "[fetch_so101_usd] already present: ${DST}"
    exit 0
fi

if [[ -n "${OPENSO101_SO101_USD_SRC:-}" ]]; then
    echo "[fetch_so101_usd] copying from \$OPENSO101_SO101_USD_SRC=${OPENSO101_SO101_USD_SRC}"
    cp "${OPENSO101_SO101_USD_SRC}" "${DST}"
elif [[ -n "${OPENSO101_SO101_USD_URL:-}" ]]; then
    echo "[fetch_so101_usd] downloading from ${OPENSO101_SO101_USD_URL}"
    curl -fsSL -o "${DST}" "${OPENSO101_SO101_USD_URL}"
elif [[ -f "${REPO_ROOT}/../safe_sim2real/outputs/third_party/so101_usd/SO-ARM101-USD.usd" ]]; then
    LEGACY="${REPO_ROOT}/../safe_sim2real/outputs/third_party/so101_usd/SO-ARM101-USD.usd"
    echo "[fetch_so101_usd] copying from sibling safe_sim2real checkout: ${LEGACY}"
    cp "${LEGACY}" "${DST}"
else
    cat >&2 <<EOF
[fetch_so101_usd] ERROR: no source for SO-ARM101-USD.usd found.

Set one of:
  OPENSO101_SO101_USD_SRC=/path/to/SO-ARM101-USD.usd      # local file
  OPENSO101_SO101_USD_URL=https://.../SO-ARM101-USD.usd   # remote URL

…or place a sibling safe_sim2real checkout at:
  ${REPO_ROOT}/../safe_sim2real/outputs/third_party/so101_usd/SO-ARM101-USD.usd

The upstream asset is authored by Lior Ben Horin (MIT-licensed); see
LICENSE-BSD-3-CLAUSE for the third-party license bundled with this repo.
EOF
    exit 1
fi

echo "[fetch_so101_usd] OK -> ${DST}"
