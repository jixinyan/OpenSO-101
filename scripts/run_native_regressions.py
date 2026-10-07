import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from openso101.rl.gpu_scope import configure_visible_gpu


configure_visible_gpu()
os.environ["OPENSO101_SKIP_ISAAC"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--worker-report", type=Path)
args = parser.parse_args()
if args.worker_report is None:
    report = Path("outputs/rl_progress") / f"native_regressions_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    process = subprocess.run([sys.executable, __file__, "--worker-report", str(report)], check=True)
    result = json.loads(report.read_text())
    print(json.dumps({"report": str(report), **result}), flush=True)
    sys.exit(result["exit_code"])

if args.worker_report.exists():
    raise FileExistsError(args.worker_report)
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import pytest

try:
    result = pytest.main([
        "tests/scenes", "tests/test_cpu_regressions.py", "tests/test_cli_rl.py",
        "-k", "not eagerly and not model_service and not agent_loop_materializes",
        "--basetemp=outputs/pytest-20261006-native", "-q",
    ])
    args.worker_report.write_text(json.dumps({"exit_code": int(result), "worker_pid": os.getpid(),
                                             "git_sha": subprocess.run(
                                                 ["git", "rev-parse", "HEAD"], check=True, text=True,
                                                 capture_output=True).stdout.strip()}, indent=2) + "\n")
finally:
    app.close()
sys.exit(result)
