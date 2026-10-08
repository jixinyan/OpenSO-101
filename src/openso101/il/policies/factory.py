# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    if not Path(path).expanduser().exists() and _looks_like_hub_repo_id(s):
        # HF Hub repo id — download to the local HF cache and return the
        # cached snapshot dir. Idempotent: subsequent calls hit the cache.
        from huggingface_hub import snapshot_download
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
) -> Any:
    """读取实际权重、配置与 observation/action processors，返回推理模型。"""
    local = Path(path).expanduser()
    if (local / "student.json").is_file():
        from openso101.rl.student import RLStudentPolicy

        return RLStudentPolicy(local.resolve(), device or "cpu")
    ckpt_dir = _resolve_checkpoint_dir(path)

    import lerobot.policies  # noqa: F401  注册实际 policy 配置类型。

    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cls = policy_class(cfg.type)
    policy = cls.from_pretrained(ckpt_dir)
    if device is not None:
        policy = policy.to(device)
    policy.eval()

    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt_dir))
    policy.openso101_preprocessor = preprocessor
    policy.openso101_postprocessor = postprocessor
    return policy


__all__ = ["load_policy", "policy_class"]
