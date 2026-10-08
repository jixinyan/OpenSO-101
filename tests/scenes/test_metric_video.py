import cv2
import numpy as np
import pytest
from pathlib import Path

from openso101.scenes.metric_video import CameraMeasurements, ImagePoint, _solve_calibration, pixel_on_plane
from openso101.scenes.models import file_digest


@pytest.fixture
def analytic_camera():
    # 数学投影用于 OpenCV 数值检查；视频与实际场景恢复分别验收。
    matrix = np.array(((500., 0, 320), (0, 500., 240), (0, 0, 1)))
    positions = np.array([(x, y, z) for z in (0., .1) for x in (-.15, 0., .15) for y in (-.1, .1)])
    pixels = cv2.projectPoints(positions, np.array((.1, .2, .05)), np.array((0., 0., 1.)), matrix,
                               np.zeros(5))[0].reshape(-1, 2)
    points = [ImagePoint(identifier=f"point_{index}", world_position_m=tuple(xyz), pixel_xy=tuple(uv))
              for index, (xyz, uv) in enumerate(zip(positions, pixels, strict=True))]
    video = Path("outputs/rl_progress/v4_seed43_policy_video/policy.mp4")
    return CameraMeasurements(video_sha256=file_digest(video), frame_index=0, image_size=(640, 480),
                              intrinsic_matrix=tuple(map(tuple, matrix)), fit_points=tuple(points[:8]),
                              validation_points=tuple(points[8:]), measurement_source="analytic_projection_check")


def test_metric_plane_projection_and_independent_numeric_points(analytic_camera):
    report = _solve_calibration(analytic_camera)
    assert report["reprojection"]["validation"]["maximum_error_px"] < 1e-6
    assert report["scene_reconstruction_verified"] is False
    point = analytic_camera.validation_points[0]
    xyz = pixel_on_plane(point.pixel_xy, point.world_position_m[2], report)
    np.testing.assert_allclose(xyz, point.world_position_m, atol=1e-8)


def test_modified_camera_pose_report_is_rejected(analytic_camera):
    report = _solve_calibration(analytic_camera)
    report["camera_position_world_m"][0] += .1
    with pytest.raises(ValueError, match="标定报告"):
        pixel_on_plane((320, 240), 0., report)


def test_reused_fit_landmark_is_rejected(analytic_camera):
    content = analytic_camera.model_dump()
    content["validation_points"][0]["identifier"] = content["fit_points"][0]["identifier"]
    with pytest.raises(ValueError, match="不同 identifier"):
        CameraMeasurements.model_validate(content)


def test_independent_reprojection_failure_is_rejected(analytic_camera):
    content = analytic_camera.model_dump()
    xy = content["validation_points"][0]["pixel_xy"]
    content["validation_points"][0]["pixel_xy"] = (xy[0] + 20., xy[1])
    with pytest.raises(ValueError, match="validation 重投影误差"):
        _solve_calibration(CameraMeasurements.model_validate(content))
