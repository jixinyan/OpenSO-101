# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False})
    from openso101.scenes.usd import compile_bundle

    compile_bundle(args.bundle, args.output)
    app.close()


if __name__ == "__main__":
    main()
