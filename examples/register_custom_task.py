# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""Register a custom OpenSO-101 task with one decorator.

The simplest way to add a task is to subclass a built-in one and tweak it in
``__post_init__``. The ``@register_task`` decorator wires it into the Gym
registry under your own ID, so ``gym.make("MyLab-PickPlace-Wide-v0")`` and the
``openso101`` CLI both see it with no framework fork.

Run (requires Isaac Sim / Isaac Lab — see docs/guides/install.md):

    PYTHONPATH=src python examples/register_custom_task.py
"""

from __future__ import annotations

from openso101.envs import register_task
from openso101.tasks.pick_place import PickPlaceEnvCfg


@register_task("MyLab-PickPlace-Wide-v0")
class WidePickPlaceEnvCfg(PickPlaceEnvCfg):
    """Pick-and-place with a wider cube-reset range (harder exploration)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Widen the cube spawn jitter from the default (+/-5 cm, +/-4 cm).
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.06, 0.06),
            "z": (0.0, 0.0),
        }


def main() -> None:
    import gymnasium as gym

    import openso101.tasks  # noqa: F401 — registers the built-in tasks

    print("Registered OpenSO-101 / custom task IDs:")
    for spec_id in sorted(gym.registry):
        if "PickPlace" in spec_id or spec_id.startswith("MyLab-"):
            print("  ", spec_id)

    # Build the custom env to confirm it constructs.
    env = gym.make("MyLab-PickPlace-Wide-v0")
    print("MyLab-PickPlace-Wide-v0 built with",
          env.unwrapped.cfg.events.reset_object_position.params["pose_range"])
    env.close()


if __name__ == "__main__":
    main()
