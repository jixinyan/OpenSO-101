from pathlib import Path

import cv2
import numpy as np
from pydantic import Field, model_validator

from .models import Digest, Identifier, Model, Pose, SceneSpec, Vector3, file_digest


class ImagePoint(Model):
    identifier: Identifier
    world_position_m: Vector3
    pixel_xy: tuple[float, float]


class CameraMeasurements(Model):
    camera_model: str = Field(default="opencv_pinhole", pattern="^opencv_pinhole$")
    video_sha256: Digest
    frame_index: int = Field(ge=0)
    image_size: tuple[int, int]
    intrinsic_matrix: tuple[Vector3, Vector3, Vector3]
    distortion: tuple[float, ...] = (0., 0., 0., 0., 0.)
    fit_points: tuple[ImagePoint, ...] = Field(min_length=6)
    validation_points: tuple[ImagePoint, ...] = Field(min_length=3)
    maximum_reprojection_error_px: float = Field(default=2., gt=0)
    measurement_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_measurements(self):
        if min(self.image_size) <= 0 or len(self.distortion) not in (4, 5, 8, 12, 14):
            raise ValueError("相机尺寸或 distortion 数量无效")
        matrix = np.asarray(self.intrinsic_matrix)
        if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1])
                or matrix[0, 1] != 0 or matrix[1, 0] != 0):
            raise ValueError("intrinsic_matrix 必须使用 OpenCV 相机坐标")
        names = [point.identifier for point in (*self.fit_points, *self.validation_points)]
        if len(names) != len(set(names)):
            raise ValueError("拟合点与独立验证点必须具有不同 identifier")
        for point in (*self.fit_points, *self.validation_points):
            if any(value < 0 or value >= maximum for value, maximum in zip(point.pixel_xy, self.image_size)):
                raise ValueError("测量点超出视频图像范围")
        coordinates = np.asarray([point.world_position_m for point in self.fit_points])
        if np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0)) < 2:
            raise ValueError("相机标定点不能共线")
        return self


class ObjectImageMeasurement(Model):
    entity_id: Identifier
    pixel_xy: tuple[float, float]
    reference_height_m: float
    measurement_source: str = Field(min_length=1)


def calibrate_camera(measurements: CameraMeasurements, video: Path) -> dict:
    import av

    if file_digest(video) != measurements.video_sha256:
        raise ValueError("相机测量与输入视频 SHA256 不一致")
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != measurements.image_size:
            raise ValueError("标定图像尺寸与视频不一致")
        frame_count = sum(1 for _ in container.decode(video=0))
    if measurements.frame_index >= frame_count:
        raise ValueError("标定帧超出实际视频范围")
    return _solve_calibration(measurements)


def _solve_calibration(measurements: CameraMeasurements) -> dict:
    matrix = np.asarray(measurements.intrinsic_matrix, dtype=np.float64)
    distortion = np.asarray(measurements.distortion, dtype=np.float64)
    objects = np.asarray([point.world_position_m for point in measurements.fit_points], dtype=np.float64)
    pixels = np.asarray([point.pixel_xy for point in measurements.fit_points], dtype=np.float64)
    successful, rvec, tvec = cv2.solvePnP(objects, pixels, matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
    if not successful:
        raise RuntimeError("OpenCV 相机标定未成功")
    rvec, tvec = cv2.solvePnPRefineLM(objects, pixels, matrix, distortion, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    errors = {}
    for name, points in (("fit", measurements.fit_points), ("validation", measurements.validation_points)):
        xyz = np.asarray([point.world_position_m for point in points], dtype=np.float64)
        uv = np.asarray([point.pixel_xy for point in points], dtype=np.float64)
        depth = (rotation @ xyz.T + tvec)[2]
        if (depth <= 0).any():
            raise ValueError("相机标定的测量点位于相机后方")
        projected = cv2.projectPoints(xyz, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
        error = np.linalg.norm(projected - uv, axis=1)
        if error.max() > measurements.maximum_reprojection_error_px:
            raise ValueError(f"相机 {name} 重投影误差超出要求：{error.max()} px")
        errors[name] = {"points": len(points), "maximum_error_px": float(error.max()),
                        "rmse_px": float(np.sqrt(np.mean(error ** 2)))}
    return {"schema_version": 1, "status": "camera_metric_calibration_verified",
            "measurements": measurements.model_dump(mode="json"), "measurements_sha256": measurements.digest(),
            "world_to_camera_rotation": rotation.tolist(), "world_to_camera_translation_m": tvec.ravel().tolist(),
            "camera_position_world_m": (-rotation.T @ tvec).ravel().tolist(),
            "reprojection": errors, "opencv_version": cv2.__version__,
            "source_sha256": file_digest(Path(__file__)), "scene_reconstruction_verified": False}


def verify_calibration(calibration: dict, video: Path | None = None) -> CameraMeasurements:
    measurements = CameraMeasurements.model_validate(calibration["measurements"])
    expected = _solve_calibration(measurements) if video is None else calibrate_camera(measurements, video)
    if calibration != expected:
        raise ValueError("标定报告与实际测量、视频或当前程序不一致")
    return measurements


def pixel_on_plane(pixel_xy, height_m, calibration: dict) -> np.ndarray:
    measurements = verify_calibration(calibration)
    pixel = np.asarray(pixel_xy, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or not np.isfinite(height_m):
        raise ValueError("像素和参考高度必须为有限数值")
    if (pixel < 0).any() or (pixel >= measurements.image_size).any():
        raise ValueError("物体测量超出图像范围")
    matrix = np.asarray(measurements.intrinsic_matrix, dtype=np.float64)
    uv = cv2.undistortPoints(pixel.reshape(1, 1, 2), matrix, np.asarray(measurements.distortion))[0, 0]
    rotation = np.asarray(calibration["world_to_camera_rotation"])
    origin = np.asarray(calibration["camera_position_world_m"])
    direction = rotation.T @ np.asarray((*uv, 1.))
    if abs(direction[2]) < 1e-9:
        raise ValueError("测量射线与参考平面平行")
    distance = (height_m - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("参考平面位于测量射线后方")
    return origin + distance * direction


def apply_metric_positions(spec: SceneSpec, observations: tuple[ObjectImageMeasurement, ...], calibration: dict):
    if not observations:
        raise ValueError("视频位置恢复需要实际物体测量")
    ids = [point.entity_id for point in observations]
    if len(ids) != len(set(ids)) or set(ids) - {entity.entity_id for entity in spec.entities}:
        raise ValueError("物体测量包含重复或未知 identifier")
    points = {point.entity_id: point for point in observations}
    entities = []
    for entity in spec.entities:
        if entity.entity_id not in points:
            entities.append(entity)
            continue
        measurement = points[entity.entity_id]
        position = pixel_on_plane(measurement.pixel_xy, measurement.reference_height_m, calibration)
        entities.append(entity.model_copy(update={"pose": Pose(position=tuple(position),
                                                               quaternion_wxyz=entity.pose.quaternion_wxyz)}))
    return SceneSpec.model_validate(spec.model_dump() | {"entities": entities})
