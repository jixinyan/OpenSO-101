# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""USD-geometry helpers for the SO-101 asset.

Lazy-imports ``pxr`` because that library is bundled inside Isaac Sim's
``extscache`` and is only importable after ``isaaclab.app.AppLauncher``
has run. When called from standalone Python (e.g. unit tests without
the sim app), the helpers fall back to a baked-in constant rather than
raising, so import-time consumers degrade gracefully.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_BASE_PRIM_NAME = "base"

# Baked from `assets/so101/usd/SO-ARM101-USD.usd` on 2026-05-14
# via the in-sim path of ``base_prim_local_z_min``. Used as a fallback when
# pxr cannot be imported (i.e. outside the Omniverse app context).
_BAKED_BASE_PRIM_LOCAL_Z_MIN: float = 0.03008


@lru_cache(maxsize=8)
def base_prim_local_z_min(usd_path: Path | str) -> float:
    """Return the z-min of the SO-101 ``base`` prim bbox in asset-local frame.

    Walks the USD looking for the first prim named ``base`` and reads its
    world-aligned bounding box via ``UsdGeom.BBoxCache``. The stage has no
    upper xforms, so "world" here equals the asset's local frame.

    If ``pxr`` cannot be imported (standalone Python without Isaac Sim
    bootstrapped), returns ``_BAKED_BASE_PRIM_LOCAL_Z_MIN`` so module-import
    consumers do not crash. Update the baked value by re-running this
    function under the Omniverse app.

    Parameters
    ----------
    usd_path:
        Path to the SO-101 USD file.

    Returns
    -------
    float
        The local z-coordinate of the lowest point of the base prim's bbox.

    Raises
    ------
    FileNotFoundError
        If pxr is available but the USD cannot be opened.
    LookupError
        If pxr is available but no prim named ``base`` is found.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return _BAKED_BASE_PRIM_LOCAL_Z_MIN

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"Could not open USD: {usd_path}")

    base_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == _BASE_PRIM_NAME:
            base_prim = prim
            break
    if base_prim is None:
        raise LookupError(
            f"No prim named {_BASE_PRIM_NAME!r} found in USD {usd_path}. "
            "Has the asset been re-exported with a different prim hierarchy?"
        )

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    bbox = cache.ComputeWorldBound(base_prim)
    return float(bbox.GetBox().GetMin()[2])


def tabletop_root_z(usd_path: Path | str) -> float:
    """Return the ``init_state.pos.z`` that places the SO-101 base bottom on world z=0.

    If the base mesh sits +k m above the articulation root, dropping the root
    by -k m puts the bottom on the table. Equivalent to
    ``-base_prim_local_z_min(usd_path)``.
    """
    return -base_prim_local_z_min(usd_path)
