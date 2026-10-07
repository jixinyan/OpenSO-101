import os
import sys

from openso101.rl.gpu_scope import configure_visible_gpu


configure_visible_gpu()
os.environ["OPENSO101_SKIP_ISAAC"] = "1"
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import pytest

try:
    result = pytest.main([
        "tests/scenes", "tests/test_cpu_regressions.py", "tests/test_cli_rl.py",
        "-k", "not eagerly and not model_service and not agent_loop_materializes",
        "--basetemp=outputs/pytest-20261006-native", "-q",
    ])
finally:
    app.close()
sys.exit(result)
