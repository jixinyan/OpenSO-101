# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""IL training entry point — programmatic wrapper over `lerobot.scripts.train`.

This module mirrors `openso101 il train`'s CLI behavior but exposes it as
a Python function so notebooks, sweep drivers, and downstream tooling can
launch training without shelling through argparse.

Why a subprocess wrapper rather than an in-process call? `lerobot.scripts.
train.train()` does heavy global setup (multiprocessing, signal handlers,
accelerator init) that does not survive being called twice in the same
interpreter; running it as a subprocess matches LeRobot's own
recommendation and gives us a clean exit code regardless of in-flight
GPU state.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_SUPPORTED_POLICIES = ("act", "diffusion", "vqbet", "tdmpc", "pi0")


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a `train_il_policy` invocation."""

    returncode: int
    output_dir: Path
    command: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def last_checkpoint(self) -> Path:
        """Conventional LeRobot last-checkpoint path under `output_dir`."""
        return self.output_dir / "checkpoints" / "last" / "pretrained_model"


def _default_output_dir(policy: str) -> Path:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs/lerobot") / f"openso101_{policy}" / stamp


def train_il_policy(
    *,
    policy: str,
    dataset: str | Path,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    wandb: bool = False,
    extra_args: Sequence[str] | None = None,
    check: bool = False,
) -> TrainResult:
    """Train an IL policy on a LeRobot dataset.

    Parameters
    ----------
    policy:
        LeRobot policy name. Tested values: ``"act"``, ``"diffusion"``.
        Other LeRobot-supported names pass through unchanged.
    dataset:
        Either a Hugging Face Hub ``user/repo`` id, or a local path to a
        LeRobot dataset directory (the kind `openso101 il record` writes).
    output_dir:
        Where LeRobot should write `checkpoints/`, `wandb/`, etc.
        Defaults to ``logs/lerobot/openso101_<policy>/<timestamp>``.
    repo_id:
        Overrides the synthetic ``local/<dataset-name>`` repo_id used when
        ``dataset`` is a local directory. Ignored for Hub datasets.
    steps, batch_size:
        Standard LeRobot trainer flags. Pass-through.
    wandb:
        Enable LeRobot's W&B logging.
    extra_args:
        Free-form CLI tokens forwarded verbatim to `lerobot.scripts.train`.
        Use this for any flag we don't surface explicitly.
    check:
        If True, raise ``RuntimeError`` on a non-zero exit instead of
        returning a failed `TrainResult`.

    Returns
    -------
    TrainResult
        Includes the resolved output dir, the exact command line, and the
        subprocess return code.
    """
    if policy not in _SUPPORTED_POLICIES:
        # Don't reject — LeRobot may grow new policies; we just warn.
        print(
            f"[openso101.il] note: policy '{policy}' is outside our tested "
            f"set {_SUPPORTED_POLICIES}; forwarding to LeRobot anyway."
        )

    dataset_path = Path(str(dataset)).expanduser()
    is_local = dataset_path.exists() and dataset_path.is_dir()

    out = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir(policy).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # LeRobot 0.4.0 renamed every script to `lerobot_<name>` and dropped the
    # short module names. The training entry point is now
    # `lerobot.scripts.lerobot_train` (or the `lerobot-train` console script).
    cmd: list[str] = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.type={policy}",
        f"--output_dir={out}",
    ]
    if is_local:
        effective_repo_id = repo_id or f"local/{dataset_path.name}"
        cmd += [
            f"--dataset.repo_id={effective_repo_id}",
            f"--dataset.root={dataset_path}",
        ]
    else:
        cmd.append(f"--dataset.repo_id={dataset}")
    if steps is not None:
        cmd.append(f"--steps={int(steps)}")
    if batch_size is not None:
        cmd.append(f"--batch_size={int(batch_size)}")
    if wandb:
        cmd.append("--wandb.enable=true")
    if extra_args:
        cmd.extend(list(extra_args))

    print(f"[openso101.il] launching: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[openso101.il] output dir: {out}")
    completed = subprocess.run(cmd)
    result = TrainResult(returncode=completed.returncode, output_dir=out, command=tuple(cmd))
    if check and not result.succeeded:
        raise RuntimeError(
            f"lerobot training exited with code {result.returncode}. "
            f"command: {' '.join(shlex.quote(c) for c in cmd)}"
        )
    return result


__all__ = ["TrainResult", "train_il_policy"]
