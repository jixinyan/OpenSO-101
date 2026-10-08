from pathlib import Path

import numpy as np
import pytest

from openso101.scenes.metric_video import CameraMeasurements, _solve_calibration, calibrate_camera, pixel_on_plane


@pytest.fixture
def measured_camera():
    source = Path(__file__).parent / "assets/metric_video"
    measurements = CameraMeasurements.model_validate_json((source / "measurements.json").read_text())
    return measurements, source / "landmarks.mp4"


def test_actual_cpu_video_and_independent_metric_points(measured_camera):
    measurements, video = measured_camera
    report = calibrate_camera(measurements, video)
    assert report["reprojection"]["validation"]["maximum_error_px"] < .3
    assert report["scene_reconstruction_verified"] is False
    for point in measurements.validation_points:
        xyz = pixel_on_plane(point.pixel_xy, point.world_position_m[2], report)
        assert np.linalg.norm(xyz - point.world_position_m) < .001


def test_modified_camera_pose_report_is_rejected(measured_camera):
    measurements, video = measured_camera
    report = calibrate_camera(measurements, video)
    report["camera_position_world_m"][0] += .1
    with pytest.raises(ValueError, match="标定报告"):
        pixel_on_plane((320, 240), 0., report)


def test_reused_fit_landmark_is_rejected(measured_camera):
    content = measured_camera[0].model_dump()
    content["validation_points"][0]["identifier"] = content["fit_points"][0]["identifier"]
    with pytest.raises(ValueError, match="不同 identifier"):
        CameraMeasurements.model_validate(content)


def test_independent_reprojection_failure_is_rejected(measured_camera):
    content = measured_camera[0].model_dump()
    xy = content["validation_points"][0]["pixel_xy"]
    content["validation_points"][0]["pixel_xy"] = (xy[0] + 20., xy[1])
    with pytest.raises(ValueError, match="validation 重投影误差"):
        _solve_calibration(CameraMeasurements.model_validate(content))
