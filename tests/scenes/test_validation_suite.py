import json
from pathlib import Path

import pytest
import trimesh

from openso101.scenes.capabilities import GeometryProbe
from openso101.scenes.catalog import AssetCatalog
from openso101.scenes.models import file_digest
from openso101.validation.suite import ValidationSuite, run_suite


@pytest.fixture
def geometry_suite(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    model = tmp_path / "block.glb"
    trimesh.creation.box(extents=(.04, .04, .04)).export(model)
    catalog = AssetCatalog(tmp_path / "assets")
    asset = catalog.import_glb(model, uid=file_digest(model)[:32], metadata={
        "name": "block", "viewerUrl": "procedural:trimesh.creation.box", "license": "MIT",
        "user": {"displayName": "OpenSO-101"},
    })
    probe = GeometryProbe(capability="graspable", dimensions_m=(.04, .04, .04), collision="convexHull",
                          maximum_gripper_opening_m=.05,
                          robot_sha256=file_digest(repo / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml"))
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(probe.model_dump_json())
    content = {
        "source_files": [str(model.relative_to(repo))],
        "stages": [
            {"name": "geometry", "resource": "cpu", "command": ["{python}", "-m", "openso101.cli.main",
                "scenes", "inspect", asset.uid, "--catalog", str(catalog.root), "--probe", str(probe_path),
                "--output", "{output}/geometry.json"],
             "requirements": [{"report": "geometry.json",
                 "pointer": "/capabilities/graspable/geometry_status", "equals": "verified"}]},
            {"name": "native", "resource": "gpu", "depends_on": ["geometry"],
             "command": ["bash", "scripts/run_native_python.sh", "2", "scripts/run_native_regressions.py",
                         "--worker-report", "{output}/native.json"],
             "requirements": [{"report": "native.json", "pointer": "/exit_code", "equals": 0}]},
        ],
    }
    manifest = tmp_path / "suite_manifest.json"
    manifest.write_text(json.dumps(content))
    return repo, manifest, tmp_path / "acceptance", model


def test_real_cpu_geometry_process_and_gpu_phase_boundary(geometry_suite):
    repo, manifest, output, _ = geometry_suite
    report = run_suite(manifest, output, phase="cpu", repo=repo)
    assert report["status"] == "cpu_verified" and report["gpu_tests_started"] is False
    assert report["stages"][0]["status"] == "verified"
    report_file = output / "geometry.json"
    content = json.loads(report_file.read_text())
    content["capabilities"]["graspable"]["geometry_status"] = "rejected"
    report_file.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="验收要求"):
        run_suite(manifest, output, phase="gpu", repo=repo)
    assert json.loads((output / "suite.json").read_text())["gpu_tests_started"] is False


def test_source_changes_prevent_gpu_execution(geometry_suite):
    repo, manifest, output, model = geometry_suite
    run_suite(manifest, output, phase="cpu", repo=repo)
    with model.open("ab") as stream:
        stream.write(b"changed source geometry")
    with pytest.raises(ValueError, match="相同源码"):
        run_suite(manifest, output, phase="gpu", repo=repo)
    assert json.loads((output / "suite.json").read_text())["gpu_tests_started"] is False


@pytest.mark.parametrize("command", [
    ["bash", "scripts/run_native_python.sh", "0", "scripts/run_native_regressions.py"],
    ["python", "scripts/run_native_regressions.py"],
    ["bash", "scripts/run_native_python.sh", "2", "-c", "print(1)"],
])
def test_uncontrolled_gpu_commands_are_rejected(geometry_suite, command):
    _, manifest, _, _ = geometry_suite
    content = json.loads(manifest.read_text())
    content["stages"][1]["command"] = command
    with pytest.raises(ValueError):
        ValidationSuite.model_validate(content)
