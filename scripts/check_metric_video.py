import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import av
import mujoco
import numpy as np

from openso101.scenes.metric_video import CameraMeasurements, ImagePoint, calibrate_camera, pixel_on_plane
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("CPU 视频检查需要 MUJOCO_GL=osmesa 与空的 CUDA_VISIBLE_DEVICES")
args.output.mkdir(parents=True, exist_ok=False)
width, height, fovy = 640, 480, 55.
root = ET.Element("mujoco")
visual = ET.SubElement(root, "visual")
ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
world = ET.SubElement(root, "worldbody")
ET.SubElement(world, "camera", name="calibration", pos="0.3 -0.5 0.4", xyaxes="1 0 0 0 0.624695 0.780869",
              fovy=str(fovy))
ET.SubElement(world, "light", pos="0.3 -0.4 0.8", dir="0 0 -1")
for index in range(12):
    position = (.14 + .105 * (index % 4), -.1 + .1 * (index // 4), .03 + .015 * (index % 5))
    color = np.asarray((.3 + .2 * (index % 4), .3 + .25 * (index // 4), .5))
    ET.SubElement(world, "geom", name=f"landmark_{index}", type="sphere", size="0.004",
                  pos=" ".join(map(str, position)), rgba=" ".join(map(str, (*color, 1.))))
model_path = args.output / "landmarks.xml"
ET.ElementTree(root).write(model_path)
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
points = []
with mujoco.Renderer(model, height=height, width=width) as renderer:
    renderer.update_scene(data, camera="calibration")
    pixels = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    segments = renderer.render()
    for index in range(12):
        geom = model.geom(f"landmark_{index}").id
        y, x = np.where((segments[:, :, 0] == geom) & (segments[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))
        if len(x) < 3:
            raise RuntimeError("CPU renderer 未提供完整的实际 landmark 图像")
        points.append(ImagePoint(identifier=f"landmark_{index}", world_position_m=tuple(data.geom_xpos[geom]),
                                 pixel_xy=(float(x.mean()), float(y.mean()))))
video = args.output / "landmarks.mp4"
with av.open(str(video), mode="w") as container:
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for _ in range(30):
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
measurements = CameraMeasurements(video_sha256=file_digest(video), frame_index=0, image_size=(width, height),
    intrinsic_matrix=((focal, 0, (width - 1) / 2), (0, focal, (height - 1) / 2), (0, 0, 1)),
    fit_points=tuple(points[:8]), validation_points=tuple(points[8:]),
    measurement_source="MuJoCo CPU OSMesa actual geom segmentation centroids")
calibration = calibrate_camera(measurements, video)
errors = [float(np.linalg.norm(pixel_on_plane(point.pixel_xy, point.world_position_m[2], calibration)
                               - point.world_position_m)) for point in points[8:]]
if max(errors) > .003:
    raise RuntimeError("CPU 视频独立物体位置恢复超过 3 mm")
with (args.output / "measurements.json").open("x") as stream:
    stream.write(measurements.model_dump_json(indent=2))
with (args.output / "calibration.json").open("x") as stream:
    json.dump(calibration, stream, ensure_ascii=False, indent=2)
report = {"status": "simulation_metric_video_verified", "scope": "MuJoCo_CPU_OSMesa_landmarks",
          "video_sha256": file_digest(video), "model_sha256": file_digest(model_path),
          "measurement_sha256": measurements.digest(), "calibration_sha256": file_digest(args.output / "calibration.json"),
          "validation_points": len(errors), "maximum_position_error_m": max(errors),
          "reprojection": calibration["reprojection"], "mujoco_version": mujoco.__version__,
          "source_sha256": file_digest(Path(__file__)), "gpu_tests_started": False,
          "real_video_reconstruction_verified": False, "robot_task_success_verified": False}
with (args.output / "report.json").open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report, indent=2))
