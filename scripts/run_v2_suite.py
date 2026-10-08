import argparse
import json
from pathlib import Path

from openso101.validation.suite import run_suite


parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--phase", choices=("cpu", "gpu"), required=True)
args = parser.parse_args()
report = run_suite(args.manifest, args.output, phase=args.phase, repo=Path(__file__).resolve().parents[1])
print(json.dumps({"status": report["status"], "stages": len(report["stages"]),
                  "gpu_tests_started": report["gpu_tests_started"], "output": str(args.output.resolve())}))
