# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Generic policy loader — wraps LeRobot's factory.

`load_policy(path)` is the single entry point used by `openso101 il play`,
`openso101 sim2real deploy`, and downstream consumers. It accepts the same
on-disk checkpoint layout that `lerobot.scripts.train` writes
(`outputs/train/<run>/checkpoints/last/pretrained_model`) as well as the
common shorthand `outputs/train/<run>` (in which case we walk down to
`checkpoints/last/pretrained_model` automatically).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_checkpoint_dir(path: str | Path) -> Path:
    """Walk down common LeRobot checkpoint layouts to the actual model dir."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"policy checkpoint not found: {p}")
    # Already pointing at a pretrained_model dir
    if (p / "config.json").exists():
        return p
    # Common shorthands LeRobot writes
    candidates = (
        p / "pretrained_model",
        p / "checkpoints" / "last" / "pretrained_model",
    )
    for c in candidates:
        if (c / "config.json").exists():
            return c
    raise FileNotFoundError(
        f"could not find a pretrained_model/config.json under {p}; "
        "expected either the model dir itself, `<run>/pretrained_model`, "
        "or `<run>/checkpoints/last/pretrained_model`."
    )


def policy_class(name: str) -> type:
    """Return the LeRobot policy class for an algorithm name (act, diffusion, ...)."""
    from lerobot.policies.factory import get_policy_class

    return get_policy_class(name)


def load_policy(path: str | Path, *, device: str | None = None) -> Any:
    """Load a LeRobot-trained policy checkpoint into an inference-ready instance.

    Two-step pattern (required because `PreTrainedPolicy` is abstract):
        1. `PreTrainedConfig.from_pretrained(path)` reads `config.json` and
           returns the concrete config object (which knows its `type`).
        2. `get_policy_class(cfg.type).from_pretrained(path)` instantiates
           the concrete subclass (ACTPolicy, DiffusionPolicy, ...).
    """
    # Resolve the path first so callers get a clear FileNotFoundError before
    # we pay the cost of importing LeRobot (and so tests can exercise the
    # path-resolution logic without LeRobot installed).
    ckpt_dir = _resolve_checkpoint_dir(path)

    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cls = policy_class(cfg.type)
    policy = cls.from_pretrained(ckpt_dir)
    if device is not None:
        policy = policy.to(device)
    policy.eval()
    return policy


__all__ = ["load_policy", "policy_class"]
