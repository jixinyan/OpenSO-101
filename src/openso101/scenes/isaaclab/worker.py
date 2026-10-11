# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from ..bundle import verify_bundle
from .preflight import validate_scene_output
from .usd import compile_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.output = validate_scene_output(args.bundle, args.output)
    verify_bundle(args.bundle)

    from openso101.rl.gpu_scope import configure_visible_gpu

    configure_visible_gpu()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False,
                         "active_gpu": 0, "physics_gpu": 0, "max_gpu_count": 1})
    try:
        compile_bundle(args.bundle, args.output)
    finally:
        app.close()


if __name__ == "__main__":
    main()
