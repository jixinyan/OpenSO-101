# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_BASE_PRIM_NAME = "base"

@lru_cache(maxsize=8)
def base_prim_local_z_min(usd_path: Path | str) -> float:
    """Return the z-min of the SO-101 ``base`` prim bbox in asset-local frame.

    Walks the USD looking for the first prim named ``base`` and reads its
    world-aligned bounding box via ``UsdGeom.BBoxCache``. The stage has no
    upper xforms, so "world" here equals the asset's local frame.

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
    from pxr import Usd, UsdGeom

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
