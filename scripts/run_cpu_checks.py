import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
environment = os.environ.copy()
environment.update(OPENSO101_SKIP_ISAAC="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH="src")
tests = ["tests/scenes/test_bundle.py", "tests/scenes/test_editing_capabilities.py", "tests/scenes/test_bddl.py",
         "tests/scenes/test_validation_suite.py", "tests/scenes/test_metric_video.py",
         "tests/scenes/test_layout_geometry.py", "tests/test_gpu_guard.py", "tests/scenes/test_usd.py"]
junit = args.output / "pytest.xml"
with (args.output / "pytest.log").open("x") as stream:
    result = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q",
                             f"--basetemp={args.output / 'pytest_files'}", f"--junitxml={junit}"],
                            env=environment, stdout=stream, stderr=subprocess.STDOUT)
root = ET.parse(junit).getroot()
suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
counts = {key: sum(int(suite.attrib[key]) for suite in suites)
          for key in ("tests", "failures", "errors", "skipped")}
verified = result.returncode == 0 and counts["tests"] >= 49 and not any(
    counts[key] for key in ("failures", "errors", "skipped"))
report = {"status": "cpu_checks_verified" if verified else "cpu_checks_failed", "exit_code": result.returncode,
          "counts": counts, "junit_sha256": file_digest(junit), "source_sha256": file_digest(Path(__file__)),
          "gpu_tests_started": False, "physical_task_verified": False}
with (args.output / "report.json").open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report))
if not verified:
    raise RuntimeError(f"CPU 检查未通过：{args.output / 'pytest.log'}")
