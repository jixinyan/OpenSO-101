# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from openso101.rl.gpu_scope import configure_visible_gpu

    configure_visible_gpu()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False,
                         "active_gpu": 0, "physics_gpu": 0, "max_gpu_count": 1})
    from .usd import compile_bundle

    try:
        compile_bundle(args.bundle, args.output)
    finally:
        app.close()


if __name__ == "__main__":
    main()
