# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Internal helper: USD bounds inspection used by safe-spawn collision setup.

SKELETON. Source: /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/_usd_bounds.py
"""

from __future__ import annotations

from pathlib import Path


def base_prim_local_z_min(usd_path: "Path | str") -> float:
    """Return the z-min of the SO-101 ``base`` prim bbox in asset-local frame.

    Falls back to a baked constant when ``pxr`` is not importable (i.e. outside
    the Omniverse app context).

    Source reference:
    /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/_usd_bounds.py
    """
    raise NotImplementedError(
        "base_prim_local_z_min not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/_usd_bounds.py"
    )


def tabletop_root_z(usd_path: "Path | str") -> float:
    """Return the ``init_state.pos.z`` that places the SO-101 base bottom on world z=0.

    Source reference:
    /data/safe_sim2real/src/safe_sim2real/robots/trs_so101/_usd_bounds.py
    """
    raise NotImplementedError(
        "tabletop_root_z not yet ported. Source reference: "
        "/data/safe_sim2real/src/safe_sim2real/robots/trs_so101/_usd_bounds.py"
    )
