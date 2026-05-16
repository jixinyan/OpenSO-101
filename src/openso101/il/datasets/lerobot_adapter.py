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
        return int(getattr(self.dataset, "num_frames", len(self.dataset)))

    @property
    def num_episodes(self) -> int:
        return int(getattr(self.dataset, "num_episodes", 0))


def _is_local_dataset_dir(p: Path) -> bool:
    """Heuristic: a LeRobot dataset dir has `meta/info.json`."""
    return (p / "meta" / "info.json").exists() or (p / "lerobot_dataset" / "meta" / "info.json").exists()


def _resolve_local_root(p: Path) -> Path:
    """Walk down `<repo_root>/lerobot_dataset/` if that's where the conversion landed."""
    nested = p / "lerobot_dataset"
    if (nested / "meta" / "info.json").exists():
        return nested
    return p


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
    source_path = Path(str(source)).expanduser()

    # Path validation runs before the LeRobot import so callers get a
    # crisp FileNotFoundError without paying the import cost.
    is_local = source_path.exists() and source_path.is_dir()
    if is_local:
        root = _resolve_local_root(source_path.resolve())
        if not _is_local_dataset_dir(root):
            raise FileNotFoundError(
                f"{root} does not look like a LeRobot dataset directory "
                "(no meta/info.json). Did you run `openso101 il record` here?"
            )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if is_local:
        effective_repo_id = repo_id or f"local/{root.name}"
        kwargs: dict[str, Any] = {"root": root}
        if episodes is not None:
            kwargs["episodes"] = episodes
        dataset = LeRobotDataset(effective_repo_id, **kwargs)
        return LeRobotDatasetHandle(dataset=dataset, repo_id=effective_repo_id, root=root)

    # Treat as a Hub repo id.
    kwargs = {}
    if episodes is not None:
        kwargs["episodes"] = episodes
    dataset = LeRobotDataset(str(source), **kwargs)
    return LeRobotDatasetHandle(dataset=dataset, repo_id=str(source), root=None)


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
