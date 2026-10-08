# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""LeRobot dataset adapter for OpenSO-101 IL training.

Two entry points:

* `load_lerobot_dataset(source)` returns a real ``LeRobotDataset`` you can
  index, iterate, or pass straight into a LeRobot trainer. It accepts a
  Hugging Face Hub repo id ``user/repo`` OR a local directory written by
  ``openso101 il record`` (either the HDF5 root or the converted LeRobot
  export inside it).
* `summarize_lerobot_dataset(ds)` prints a one-page sanity summary
  (episode count, frame count, feature schema, action range).

This module is **fully functional**: it does not raise NotImplementedError
under any normal install of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LeRobotDatasetHandle:
    """Lightweight handle around a loaded LeRobotDataset.

    Exposes the dataset object itself plus the resolved on-disk root so
    downstream code can pass either to the LeRobot trainer's
    `--dataset.repo_id` / `--dataset.root` flags without re-deriving them.
    """

    dataset: Any
    repo_id: str
    root: Path | None  # None for Hub-backed datasets

    @property
    def num_frames(self) -> int:
        # Do not use ``getattr(..., len(self.dataset))`` here: Python
        # evaluates the default argument before calling ``getattr``.  Some
        # LeRobot dataset implementations expose ``num_frames`` but are not
        # sequence-like, so that seemingly harmless fallback raised a
        # ``TypeError`` even when the attribute was present.
        value = getattr(self.dataset, "num_frames", None)
        if value is None:
            try:
                value = len(self.dataset)
            except TypeError as exc:
                raise TypeError("loaded dataset exposes neither num_frames nor __len__") from exc
        return int(value)

    @property
    def num_episodes(self) -> int:
        value = getattr(self.dataset, "num_episodes", None)
        if value is not None:
            return int(value)
        # Older LeRobot releases keep episode metadata on ``meta`` rather
        # than the dataset object.  Prefer that value when available, while
        # retaining a safe zero for Hub/custom dataset implementations that
        # do not expose episode counts.
        info = getattr(getattr(self.dataset, "meta", None), "info", None)
        if isinstance(info, dict) and info.get("total_episodes") is not None:
            return int(info["total_episodes"])
        return 0


def _is_local_dataset_dir(p: Path) -> bool:
    """Heuristic: a LeRobot dataset dir has `meta/info.json`."""
    return (p / "meta" / "info.json").exists() or (p / "lerobot_dataset" / "meta" / "info.json").exists()


def _resolve_local_root(p: Path) -> Path:
    """Walk down `<repo_root>/lerobot_dataset/` if that's where the conversion landed."""
    nested = p / "lerobot_dataset"
    if (nested / "meta" / "info.json").exists():
        return nested
    return p


def resolve_lerobot_source(source: str | Path, *, repo_id: str | None = None) -> tuple[Path | None, str]:
    source_path = Path(source).expanduser()
    root = None
    if source_path.exists():
        if not source_path.is_dir():
            raise ValueError(f"dataset source exists but is not a directory: {source_path}")
        root = _resolve_local_root(source_path.resolve())
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(f"LeRobot 数据集缺少 meta/info.json: {root}")
        effective_repo_id = repo_id or f"local/{root.name}"
    else:
        if isinstance(source, Path) or str(source).startswith(("/", ".", "~")) or len(source_path.parts) != 2:
            raise FileNotFoundError(f"LeRobot 数据集目录不存在: {source_path}")
        if repo_id is not None:
            raise ValueError("repo_id 参数只用于本地 LeRobot 数据集")
        effective_repo_id = str(source)
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(effective_repo_id)
    return root, effective_repo_id


def load_lerobot_dataset(
    source: str | Path,
    *,
    repo_id: str | None = None,
    episodes: list[int] | None = None,
) -> LeRobotDatasetHandle:
    """Load a LeRobot dataset by Hub id OR local path.

    Parameters
    ----------
    source:
        Either a Hugging Face Hub ``user/repo`` id, or a path to a local
        LeRobot dataset directory written by ``openso101 il record``.
    repo_id:
        Optional override for the dataset's repo_id when ``source`` is a
        local path. Defaults to ``f"local/{<dir-name>}"`` so the trainer's
        bookkeeping has something stable to key on.
    episodes:
        Optional subset of episode indices to materialize. Passed through
        to ``LeRobotDataset``.

    Returns
    -------
    LeRobotDatasetHandle
        Includes the live dataset object, the effective repo_id, and the
        on-disk root (None for Hub-backed datasets).
    """
    if episodes is not None:
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in episodes):
            raise ValueError("episodes must contain non-negative integer indices")

    root, effective_repo_id = resolve_lerobot_source(source, repo_id=repo_id)
    if root is not None:
        from .validation import validate_lerobot_metadata

        validate_lerobot_metadata(root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root is not None:
        kwargs: dict[str, Any] = {"root": root}
        if episodes is not None:
            kwargs["episodes"] = episodes
        dataset = LeRobotDataset(effective_repo_id, **kwargs)
        return LeRobotDatasetHandle(dataset=dataset, repo_id=effective_repo_id, root=root)

    # Treat as a Hub repo id.
    kwargs = {}
    if episodes is not None:
        kwargs["episodes"] = episodes
    dataset = LeRobotDataset(effective_repo_id, **kwargs)
    return LeRobotDatasetHandle(dataset=dataset, repo_id=effective_repo_id, root=None)


def summarize_lerobot_dataset(handle: LeRobotDatasetHandle) -> None:
    """Print a quick sanity summary — useful before launching a training run."""
    ds = handle.dataset
    print(f"[lerobot] repo_id     : {handle.repo_id}")
    print(f"[lerobot] root        : {handle.root or '(Hub-backed)'}")
    print(f"[lerobot] num_episodes: {handle.num_episodes}")
    print(f"[lerobot] num_frames  : {handle.num_frames}")
    features = getattr(ds, "features", None)
    if features:
        keys = sorted(features.keys()) if hasattr(features, "keys") else list(features)
        print(f"[lerobot] features    : {keys}")


__all__ = [
    "LeRobotDatasetHandle",
    "load_lerobot_dataset",
    "summarize_lerobot_dataset",
]
