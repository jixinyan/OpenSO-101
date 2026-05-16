# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""ACT (Action Chunking Transformer) — thin wrapper over LeRobot.

LeRobot ships a maintained ACT implementation that already supports
SO-101-style joint-position action spaces and the observation layout
emitted by `openso101 il record`. We re-export it under
`openso101.il.policies.ACTPolicy` so:

* `from openso101.il.policies import ACTPolicy` works without coupling
  consumers to LeRobot's internal module layout (which has churned across
  versions);
* future SO-101-specific overrides can be layered in this module without
  touching call sites.

This module is **fully functional**: calling `ACTPolicy(config)` builds a
real, trainable PyTorch model; `load_act_policy(path)` loads a checkpoint
produced by `lerobot.scripts.train --policy.type act`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_act_class() -> type:
    """Resolve LeRobot's ACTPolicy class without hard-coding a fragile path."""
    # The maintained, stable entry point. We go through the factory to stay
    # consistent with the loading paths used in cli/il.py and sim2real/deploy.py.
    from lerobot.policies.factory import get_policy_class

    return get_policy_class("act")


class _ACTProxy:
    """Lazy proxy so importing this module does not require LeRobot at import-time.

    Direct instantiation (`ACTPolicy(cfg)`) resolves the real class on the
    first call and forwards. `isinstance(p, ACTPolicy)` and subclassing both
    work because we set `__class__` to the resolved type after the proxy
    forwards construction.
    """

    def __new__(cls, *args, **kwargs):
        real_cls = _import_act_class()
        return real_cls(*args, **kwargs)


ACTPolicy: Any = _ACTProxy


def load_act_policy(path: str | Path, *, device: str | None = None):
    """Load an ACT checkpoint trained by LeRobot.

    Thin wrapper over the generic loader; the algorithm name is read from
    the checkpoint's `config.json` so this also accepts non-ACT paths
    (it'll just return whatever was trained there).
    """
    from .factory import load_policy

    return load_policy(path, device=device)


__all__ = ["ACTPolicy", "load_act_policy"]
