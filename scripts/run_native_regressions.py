import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from openso101.rl.gpu_scope import configure_visible_gpu
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--worker-report", type=Path)
args = parser.parse_args()
configure_visible_gpu()
os.environ["OPENSO101_SKIP_ISAAC"] = "1"
if args.worker_report is None:
    report = Path("outputs/rl_progress") / f"native_regressions_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    process = subprocess.run([sys.executable, __file__, "--worker-report", str(report)], check=True)
    result = json.loads(report.read_text())
    print(json.dumps({"report": str(report), **result}), flush=True)
    sys.exit(result["exit_code"])

if args.worker_report.exists():
    raise FileExistsError(args.worker_report)
args.worker_report.parent.mkdir(parents=True, exist_ok=True)
git_sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True,
                         capture_output=True).stdout.strip()
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

try:
    import pytest

    junit = args.worker_report.with_suffix(".xml")
    result = pytest.main([
        "tests/scenes/test_usd.py", "tests/scenes/test_terminal_keyboard.py", "tests/scenes/test_training_config.py",
        f"--basetemp={args.worker_report.with_suffix('.pytest_files')}", f"--junitxml={junit}", "-q",
    ])
    root = ET.parse(junit).getroot()
    counts = {key: sum(int(suite.attrib[key]) for suite in root.findall("testsuite"))
              for key in ("tests", "errors", "failures", "skipped")}
    verified = not result and counts["tests"] == 8 and not any(
        counts[key] for key in ("errors", "failures", "skipped"))
    args.worker_report.write_text(json.dumps({"status": "native_program_checks_verified" if verified else "failed",
        "exit_code": int(result), "worker_pid": os.getpid(), "git_sha": git_sha, "counts": counts,
        "junit_sha256": file_digest(junit), "source_sha256": file_digest(Path(__file__)),
        "scope": "Isaac_USD_PTY_and_checkpoint_program_checks", "physical_task_success_verified": False}, indent=2) + "\n")
    if not verified:
        raise RuntimeError("原生程序检查的数量或结果未达到要求")
finally:
    app.close()
sys.exit(result)
