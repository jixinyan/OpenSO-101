import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("CPU 接口检查需要禁止 CUDA")
args.output.mkdir(parents=True, exist_ok=False)
junit = args.output / "pytest.xml"
with (args.output / "pytest.log").open("x") as stream:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/scenes/test_terminal_keyboard.py",
                             "tests/scenes/test_controls.py", "tests/scenes/test_training_config.py",
                             "tests/scenes/test_evaluation.py", "tests/test_cli_rl.py", "tests/test_cli_il.py",
                             "tests/test_lerobot_push_dataset.py", "-q",
                             "tests/scenes/test_recording_timing.py",
                             f"--basetemp={args.output / 'pytest_files'}", f"--junitxml={junit}"],
                            stdout=stream, stderr=subprocess.STDOUT)
root = ElementTree.parse(junit).getroot()
counts = {key: sum(int(suite.attrib[key]) for suite in root.findall("testsuite"))
          for key in ("tests", "errors", "failures", "skipped")}
if result.returncode or counts["tests"] != 65 or any(counts[key] for key in ("errors", "failures", "skipped")):
    raise RuntimeError("实际 PTY、控制、配置、RL/IL 命令与评估接口检查未通过")
report = {"status": "actual_cpu_interfaces_verified", "counts": counts,
          "junit_sha256": file_digest(junit), "source_sha256": file_digest(Path(__file__)),
          "gpu_tests_started": False, "physics_keyboard_collection_verified": False}
(args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report))
