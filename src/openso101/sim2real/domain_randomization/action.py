# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Action-space domain randomization: action LATENCY.

The dominant unmodeled effect for the Feetech STS3215 serial servos on the
SO-101 is **actuation latency** — the gap between a commanded joint target
and the servo actually starting to track it (serial bus round-trip + driver
queueing + servo controller lag). A policy trained without latency learns a
reaction timing the real bus can't honor, which is a common sim2real failure
mode. This module injects a bounded, randomized delay on the commanded joint
targets so the policy is forced to be robust to it.

Two complementary cores are provided, both as **pure tensor logic** (no
Isaac Lab dependency) so they can be unit-tested without booting Isaac Sim:

  * :class:`ActionDelayBuffer` — a per-env ring buffer that holds/lags the
    commanded targets by an integer number of control steps, randomized in
    ``[0, max_delay]`` per env on reset. This is the discrete "the command
    you send now is applied ``d`` steps later" model.
  * :class:`FirstOrderActionLag` — a continuous first-order lag
    ``y += alpha * (u - y)`` with a per-env randomized time constant. This is
    the smooth "the servo eases toward the command" model.

Both are **OPT-IN**. They are deliberately NOT auto-wired into any task: the
delay magnitude is a hypothesis that needs a real bus-latency measurement and
a training run to tune (a too-large delay destabilizes PPO). See
:func:`attach_action_dr` for the recommended wiring.

Recommended usage (manual wiring inside an ActionTerm or a post-physics
callback that owns the commanded targets)::

    from openso101.sim2real.domain_randomization.action import ActionDelayBuffer

    # On env construction (num_envs envs, 6 joints):
    delay = ActionDelayBuffer(num_envs=N, action_dim=6, max_delay=2, device=dev)

    # On reset (optionally pass env ids to re-randomize a subset):
    delay.reset()

    # Every control step, BEFORE the targets reach the articulation:
    applied_targets = delay.step(commanded_targets)

TUNING: ``max_delay`` is in CONTROL STEPS. At a 30 Hz control rate, a
``max_delay`` of 1-2 steps (~33-66 ms) is a reasonable starting hypothesis
for the Feetech bus, but the real value MUST be measured on hardware and
validated with a training run before relying on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


class ActionDelayBuffer:
    """Ring-buffer delay: apply the command issued ``d`` steps ago.

    Holds the last ``max_delay + 1`` commanded action tensors per env and,
    on each :meth:`step`, returns the command from ``delay_steps[env]`` steps
    in the past. ``delay_steps`` is randomized per env in ``[0, max_delay]``
    on :meth:`reset`. A delay of ``0`` is pass-through (returns the current
    command), so ``max_delay=0`` disables the effect.

    Pure tensor logic — no Isaac Lab import — so it is unit-testable.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        max_delay: int,
        *,
        device: "torch.device | str | None" = None,
        seed: int | None = None,
    ) -> None:
        import torch

        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {max_delay}")

        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.max_delay = int(max_delay)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

        # History holds the last (max_delay + 1) commands. Column 0 is the
        # most-recent command; column k is the command from k steps ago.
        self._history = torch.zeros(
            (self.num_envs, self.max_delay + 1, self.action_dim),
            device=self.device,
        )
        # Per-env integer delay in [0, max_delay].
        self._delay_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._initialized = False

    def reset(self, env_ids: "Sequence[int] | torch.Tensor | None" = None) -> None:
        """Re-randomize the per-env delay and clear history for those envs.

        ``env_ids=None`` resets all envs. Each selected env draws a fresh
        integer delay uniformly in ``[0, max_delay]``.
        """
        import torch

        if env_ids is None:
            idx = torch.arange(self.num_envs, device=self.device)
        else:
            idx = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if self.max_delay > 0:
            new_delays = torch.randint(
                low=0,
                high=self.max_delay + 1,
                size=(idx.numel(),),
                generator=self._generator,
                device=self.device,
            )
        else:
            new_delays = torch.zeros(idx.numel(), dtype=torch.long, device=self.device)
        self._delay_steps[idx] = new_delays
        # Clear history for reset envs so a stale command from a prior episode
        # cannot leak across the reset boundary.
        self._history[idx] = 0.0
        self._initialized = False

    def step(self, command: "torch.Tensor") -> "torch.Tensor":
        """Push ``command`` (shape ``[num_envs, action_dim]``) and return the delayed command.

        On the very first step after a reset (or construction) the history is
        seeded with ``command`` so the delayed read does not return zeros.
        """
        import torch

        command = torch.as_tensor(command, device=self.device, dtype=self._history.dtype)
        if command.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"command shape {tuple(command.shape)} != expected "
                f"{(self.num_envs, self.action_dim)}"
            )

        if not self._initialized:
            # Seed all history slots with the first command so the warm-up
            # period returns sensible values rather than zeros.
            self._history[:] = command.unsqueeze(1)
            self._initialized = True
        else:
            # Shift history right by one slot (slot k now holds the command
            # from k steps ago), then write the new command at slot 0.
            self._history = torch.roll(self._history, shifts=1, dims=1)
            self._history[:, 0, :] = command

        # Gather, per env, the command from delay_steps[env] slots back.
        gather_idx = self._delay_steps.view(self.num_envs, 1, 1).expand(
            self.num_envs, 1, self.action_dim
        )
        delayed = torch.gather(self._history, 1, gather_idx).squeeze(1)
        return delayed

    @property
    def delay_steps(self) -> "torch.Tensor":
        """Current per-env integer delays (read-only view)."""
        return self._delay_steps


class FirstOrderActionLag:
    """First-order lag with a per-env randomized time constant.

    Models ``y[t] = y[t-1] + alpha * (u[t] - y[t-1])`` where ``alpha`` in
    ``(0, 1]`` is drawn per env in ``[alpha_min, alpha_max]`` on :meth:`reset`.
    Smaller ``alpha`` = slower servo = more lag. ``alpha=1`` is pass-through.

    Pure tensor logic — no Isaac Lab import — so it is unit-testable.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        alpha_min: float = 0.3,
        alpha_max: float = 1.0,
        device: "torch.device | str | None" = None,
        seed: int | None = None,
    ) -> None:
        import torch

        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not (0.0 < alpha_min <= alpha_max <= 1.0):
            raise ValueError(
                "require 0 < alpha_min <= alpha_max <= 1, got "
                f"alpha_min={alpha_min}, alpha_max={alpha_max}"
            )

        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

        self._state = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._alpha = torch.full((self.num_envs, 1), self.alpha_max, device=self.device)
        self._initialized = False

    def reset(self, env_ids: "Sequence[int] | torch.Tensor | None" = None) -> None:
        """Re-randomize per-env ``alpha`` and clear lag state for those envs."""
        import torch

        if env_ids is None:
            idx = torch.arange(self.num_envs, device=self.device)
        else:
            idx = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        rand = torch.rand(
            (idx.numel(), 1), generator=self._generator, device=self.device
        )
        self._alpha[idx] = self.alpha_min + rand * (self.alpha_max - self.alpha_min)
        self._state[idx] = 0.0
        self._initialized = False

    def step(self, command: "torch.Tensor") -> "torch.Tensor":
        """Apply the first-order lag and return the lagged command."""
        import torch

        command = torch.as_tensor(command, device=self.device, dtype=self._state.dtype)
        if command.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"command shape {tuple(command.shape)} != expected "
                f"{(self.num_envs, self.action_dim)}"
            )
        if not self._initialized:
            # Seed the lag state at the first command so the first output is
            # not pulled toward zero from an arbitrary starting pose.
            self._state[:] = command
            self._initialized = True
        else:
            self._state = self._state + self._alpha * (command - self._state)
        return self._state.clone()

    @property
    def alpha(self) -> "torch.Tensor":
        """Current per-env lag coefficients (read-only view)."""
        return self._alpha


def attach_action_dr(
    env: object | None = None,
    *,
    max_delay: int = 0,
    mode: str = "delay",
    **kwargs: object,
):
    """Build an opt-in action-latency DR helper for manual wiring.

    This intentionally does NOT mutate ``env`` or auto-register an Isaac Lab
    ``EventTerm``/``ActionTerm``: where to inject the delay depends on the
    task's action pipeline, and the delay magnitude needs a hardware
    measurement plus a training run to tune (see module docstring). Wiring it
    blindly into a task would risk destabilizing PPO with an unvalidated
    delay, so we return a ready-to-use core instead and leave the placement
    to the integrator.

    Parameters
    ----------
    env:
        Optional handle, used only to infer ``num_envs``/``device`` when not
        passed explicitly via ``kwargs``. May be ``None`` for pure-logic use.
    max_delay:
        Maximum integer control-step delay for ``mode="delay"``. ``0`` (the
        default) yields a pass-through buffer — the caller must set this to a
        hardware-measured value to actually inject latency.
    mode:
        ``"delay"`` -> :class:`ActionDelayBuffer`; ``"lag"`` ->
        :class:`FirstOrderActionLag`.
    kwargs:
        Forwarded to the selected core's constructor (e.g. ``num_envs``,
        ``action_dim``, ``device``, ``seed``, ``alpha_min``, ``alpha_max``).

    Returns
    -------
    ActionDelayBuffer | FirstOrderActionLag
        A constructed, ready-to-use latency core. Call ``.reset()`` on env
        reset and wrap your commanded targets with ``.step(targets)`` BEFORE
        they reach the articulation.

    Recommended wiring (pseudo-code, inside the code that owns the targets)::

        core = attach_action_dr(env, mode="delay", max_delay=2,
                                num_envs=N, action_dim=6, device=dev)
        # on reset:        core.reset(env_ids)
        # every step:      targets = core.step(targets)

    NOTE: the recommended ``max_delay`` / ``alpha`` values are HYPOTHESES;
    they need a real hardware latency measurement and a training run to tune.
    """
    num_envs = kwargs.pop("num_envs", None)
    action_dim = kwargs.pop("action_dim", None)
    device = kwargs.pop("device", None)

    if num_envs is None and env is not None:
        num_envs = getattr(env, "num_envs", None)
    if device is None and env is not None:
        device = getattr(env, "device", None)
    if num_envs is None:
        raise ValueError(
            "attach_action_dr needs num_envs (pass num_envs=... or an env "
            "exposing .num_envs)."
        )
    if action_dim is None:
        # SO-101 has 6 controllable joints (5 arm + gripper).
        action_dim = 6

    if mode == "delay":
        return ActionDelayBuffer(
            num_envs=int(num_envs),
            action_dim=int(action_dim),
            max_delay=int(max_delay),
            device=device,
            **kwargs,
        )
    if mode == "lag":
        return FirstOrderActionLag(
            num_envs=int(num_envs),
            action_dim=int(action_dim),
            device=device,
            **kwargs,
        )
    raise ValueError(f"unknown mode {mode!r}; expected 'delay' or 'lag'")


__all__ = [
    "ActionDelayBuffer",
    "FirstOrderActionLag",
    "attach_action_dr",
]
