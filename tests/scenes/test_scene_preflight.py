import pytest

from openso101.scenes.isaaclab.preflight import (
    validate_runtime_output,
    validate_runtime_parameters,
    validate_scene_output,
)
from openso101.scenes.isaaclab.preparation import prepare_scene


@pytest.mark.parametrize("name", ("num_envs", "steps", "resets"))
@pytest.mark.parametrize("value", (True, False, 0, -1, 1.5, "4", None))
def test_invalid_counts_create_no_output(tmp_path, name, value):
    options = {"num_envs": 4, "steps": 200, "resets": 100, name: value}
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=name):
        prepare_scene(tmp_path / "unavailable_bundle", output, **options)
    assert not output.exists()


@pytest.mark.parametrize("value", (True, False, 0, -1, float("inf"), float("nan"), "1"))
def test_invalid_timeout_creates_no_output(tmp_path, value):
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="timeout_seconds"):
        prepare_scene(tmp_path / "unavailable_bundle", output, timeout_seconds=value)
    assert not output.exists()


@pytest.mark.parametrize("timeout", (None, 1, .5))
def test_valid_counts_and_timeout(timeout):
    validate_runtime_parameters(num_envs=4, steps=200, resets=100, timeout_seconds=timeout)


def test_output_is_independent_and_resolved(tmp_path):
    source = tmp_path / "scene"
    source.mkdir()
    output = tmp_path / "results" / "prepared"
    assert validate_scene_output(source, output) == output.resolve()
    assert not output.parent.exists()
    for invalid in (source, source / "generated", tmp_path):
        with pytest.raises(ValueError, match="独立"):
            validate_scene_output(source, invalid)


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "runtime.json"
    output.write_text("existing report")
    with pytest.raises(FileExistsError):
        validate_scene_output(tmp_path / "scene", output)
    assert output.read_text() == "existing report"


def test_dangling_symlink_is_preserved(tmp_path):
    output = tmp_path / "output"
    target = tmp_path / "absent"
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        validate_scene_output(tmp_path / "scene", output)
    assert output.is_symlink()
    assert output.readlink() == target
    assert not target.exists()


def test_output_parent_must_be_directory(tmp_path):
    parent = tmp_path / "parent"
    parent.write_text("existing file")
    with pytest.raises(NotADirectoryError):
        validate_scene_output(tmp_path / "scene", parent / "nested" / "output")
    assert parent.read_text() == "existing file"


def test_runtime_report_and_trace_are_independent(tmp_path):
    scene = tmp_path / "scene"
    report = tmp_path / "runtime.json"
    assert validate_runtime_output(scene, report) == report.resolve()
    with pytest.raises(ValueError, match="HDF5"):
        validate_runtime_output(scene, tmp_path / "runtime.hdf5")
    trace = report.with_suffix(".hdf5")
    trace.write_bytes(b"existing trace")
    with pytest.raises(FileExistsError):
        validate_runtime_output(scene, report)
    assert trace.read_bytes() == b"existing trace"
    assert not report.exists()
