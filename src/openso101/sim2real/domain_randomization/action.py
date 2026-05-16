# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Action-space domain randomization (intentionally deferred).

Implementing action-side DR (Gaussian noise on the commanded action,
sim-only latency injection) requires wrapping the action term itself,
which Isaac Lab doesn't expose as a config-time hook in 2.3.0. The
clean implementation needs a custom ``ActionTerm`` subclass that
intercepts ``process_actions``. Marked deferred until a concrete
sim2real transfer experiment requires it.

If you reach for this and it's still a stub, the upstream physics DR
(:mod:`openso101.sim2real.domain_randomization.physics`),
observation DR (:mod:`...domain_randomization.observation`), and
visual DR (:mod:`...domain_randomization.visual`) modules together
cover the most common sim2real failure modes; action noise is rarely
the dominant factor.
"""

from __future__ import annotations


def attach_action_dr(*args, **kwargs):
    """Placeholder — not implemented; see module docstring."""
    raise NotImplementedError(
        "Action-space DR is not implemented. Use the physics + observation + "
        "visual DR modules under openso101.sim2real.domain_randomization for "
        "the bulk of sim2real coverage."
    )


__all__ = ["attach_action_dr"]
