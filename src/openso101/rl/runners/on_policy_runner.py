# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""On-policy runner — thin wrapper over rsl_rl's OnPolicyRunner."""

from __future__ import annotations

try:
    from rsl_rl.runners import OnPolicyRunner as _RslRlRunner
except ImportError:
    _RslRlRunner = None  # type: ignore[assignment]


if _RslRlRunner is not None:
    class OnPolicyRunner(_RslRlRunner):
        """Alias for `rsl_rl.runners.OnPolicyRunner`."""
else:
    class OnPolicyRunner:
        """rsl_rl not installed — OnPolicyRunner unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "rsl_rl is required for openso101.rl.runners.OnPolicyRunner."
            )


__all__ = ["OnPolicyRunner"]
