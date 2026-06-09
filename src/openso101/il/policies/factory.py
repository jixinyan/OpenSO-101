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

import os
from pathlib import Path
from typing import Any


# Env-var escape hatch mirroring the ``allow_unnormalized`` kwarg. Set
# ``OPENSO101_ALLOW_UNNORMALIZED=1`` to downgrade a missing-processor hard
# error back to a WARN (e.g. for an unpowered dry run or a very old
# checkpoint). Off by default so play/deploy fail loudly when normalization
# stats are missing.
_ALLOW_UNNORMALIZED_ENV = "OPENSO101_ALLOW_UNNORMALIZED"


def _env_allows_unnormalized() -> bool:
    return os.environ.get(_ALLOW_UNNORMALIZED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _looks_like_hub_repo_id(s: str) -> bool:
    """Heuristic: 'user/repo' or 'org/repo' Hub identifier vs. a local path.

    HF Hub repo IDs are exactly one `/` separator with no leading/trailing
    slash and no `.` or `..` segments. Filesystem paths typically have a
    leading slash, a `.` segment, or more than one `/`. This is the same
    detection LeRobot's own from_pretrained uses internally.
    """
    if "/" not in s:
        return False
    if s.startswith(("/", "./", "../", "~")):
        return False
    parts = s.split("/")
    if len(parts) != 2:
        return False
    return all(parts) and not any(p in (".", "..") for p in parts)


def _resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve a local checkpoint path OR a HF Hub repo id to a local model dir.

    Accepts:
      * Path to a `pretrained_model/` dir directly (has config.json).
      * Path to a run dir (`<run>/pretrained_model` or
        `<run>/checkpoints/last/pretrained_model` exist under it).
      * A HF Hub repo id like `kevin831/openso101-act-pickplace-v1` — in
        which case we use huggingface_hub to download the snapshot to the
        local HF cache and return the snapshot dir.
    """
    s = str(path)
    if _looks_like_hub_repo_id(s):
        # HF Hub repo id — download to the local HF cache and return the
        # cached snapshot dir. Idempotent: subsequent calls hit the cache.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                f"'{s}' looks like a HF Hub repo id but huggingface_hub is "
                "not installed; either install it or pass a local path."
            ) from exc
        snapshot = Path(snapshot_download(repo_id=s, repo_type="model"))
        if not (snapshot / "config.json").exists():
            raise FileNotFoundError(
                f"downloaded Hub repo {s} has no config.json at {snapshot}; "
                "is this actually a LeRobot policy checkpoint?"
            )
        return snapshot

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


def load_policy(
    path: str | Path,
    *,
    device: str | None = None,
    allow_unnormalized: bool = False,
) -> Any:
    """Load a LeRobot-trained policy checkpoint into an inference-ready instance.

    Two-step pattern (required because `PreTrainedPolicy` is abstract):
        1. `PreTrainedConfig.from_pretrained(path)` reads `config.json` and
           returns the concrete config object (which knows its `type`).
        2. `get_policy_class(cfg.type).from_pretrained(path)` instantiates
           the concrete subclass (ACTPolicy, DiffusionPolicy, ...).

    By default, failing to load the pre/post-processor pipelines (which carry
    the dataset normalization stats) is a HARD ERROR: a policy without them
    consumes / emits N(0, 1) normalized values, which on hardware produces
    near-stationary or drifting behavior — a silent, dangerous failure. Pass
    ``allow_unnormalized=True`` (or set ``OPENSO101_ALLOW_UNNORMALIZED=1``) to
    downgrade the error to a WARN and return a policy with
    ``openso101_preprocessor``/``openso101_postprocessor`` set to ``None`` —
    only appropriate for an unpowered dry run or a legacy checkpoint.
    """
    # Resolve the path first so callers get a clear FileNotFoundError before
    # we pay the cost of importing LeRobot (and so tests can exercise the
    # path-resolution logic without LeRobot installed).
    ckpt_dir = _resolve_checkpoint_dir(path)

    # Importing `lerobot.policies` triggers the @register_subclass calls in
    # each policy's configuration module, which is what populates the draccus
    # choice registry that PreTrainedConfig.from_pretrained reads. Without
    # this, `from_pretrained` raises
    # `DecodingError: Couldn't find a choice class for 'act'` because the
    # subclass was never imported. LeRobot's own train script does this
    # transitively; our `il play` doesn't, so we need to be explicit.
    import lerobot.policies  # noqa: F401 — for side-effect registration

    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cls = policy_class(cfg.type)
    policy = cls.from_pretrained(ckpt_dir)
    if device is not None:
        policy = policy.to(device)
    policy.eval()

    # LeRobot's `policy.select_action(obs)` returns NORMALIZED actions
    # (and expects NORMALIZED observations) — the normalization stats live
    # in separate processor pipelines that the saved checkpoint stores
    # alongside the model weights. Without applying them, the env sees the
    # raw model output in N(0, 1) space and the arm barely moves.
    # Load the pre/post-processor pipelines and stash them on the policy
    # object so the caller (e.g. `il play`) can apply them.
    try:
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(ckpt_dir),
        )
        # Stash on the policy so callers don't need to thread three return
        # values through their loop. They're discoverable via the attrs.
        policy.openso101_preprocessor = preprocessor
        policy.openso101_postprocessor = postprocessor
    except Exception as exc:
        # If the processors can't be loaded (very old checkpoint format,
        # missing files, etc.) this is a HARD ERROR by default: without the
        # normalization stats the policy emits N(0, 1) values and the arm
        # barely moves — a silent failure that is especially dangerous on
        # real hardware. The opt-in escape hatch (allow_unnormalized kwarg or
        # OPENSO101_ALLOW_UNNORMALIZED=1) downgrades it to a WARN.
        message = (
            f"Could not load pre/post processors from {ckpt_dir}: "
            f"{type(exc).__name__}: {exc}. Policy actions would be in raw "
            "(normalized) units, which usually produces near-stationary "
            "behavior. Verify the checkpoint contains "
            "policy_preprocessor.json + policy_postprocessor.json."
        )
        if not (allow_unnormalized or _env_allows_unnormalized()):
            raise RuntimeError(
                f"[ERROR]: {message} Refusing to return an unnormalized "
                "policy. Pass allow_unnormalized=True or set "
                f"{_ALLOW_UNNORMALIZED_ENV}=1 to override (dry runs only)."
            ) from exc
        print(
            f"[WARN]: {message} Continuing with an UNNORMALIZED policy "
            f"because allow_unnormalized was requested ({_ALLOW_UNNORMALIZED_ENV} "
            "or kwarg)."
        )
        policy.openso101_preprocessor = None
        policy.openso101_postprocessor = None

    return policy


__all__ = ["load_policy", "policy_class"]
