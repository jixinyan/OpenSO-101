import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openso101.rl.config import digest
from openso101.sim2sim.mujoco import JOINT_NAMES, JOINT_OFFSETS, build_model


parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--collision-bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root = Path(__file__).resolve().parents[1]
folder = args.trajectory.parent
report = json.loads((folder / "report.json").read_text())
plan = json.loads((folder / "plan.json").read_text())
if (digest(args.trajectory) != report["trajectory_sha256"]
        or digest(folder / "plan.json") != report["plan_sha256"]
        or digest(args.collision_bundle / "manifest.json") != report["collision_bundle_sha256"]):
    raise ValueError("视频需要通过来源校验的真实夹爪轨迹和碰撞部件")
metadata = json.loads((root / "outputs/rl_progress/lift_scene_geometry_oriented_verified/policy.json").read_text())
model = build_model(root / "outputs/so-arm100/Simulation/SO101/so101_old_calib.xml", metadata, args.collision_bundle)
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720
model.geom_rgba[model.geom("object").id] = [.1, .55, .95, 1]
model.geom_rgba[model.geom("table").id] = [.65, .65, .68, 1]
data = mujoco.MjData(model)
qids = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
oq = int(model.joint("object_free").qposadr[0])
gid = model.body("gripper").id
camera = mujoco.MjvCamera()
camera.lookat[:] = [.02, -.15, .11]
camera.distance = .7
camera.azimuth = 145
camera.elevation = -25
option = mujoco.MjvOption()
option.geomgroup[3] = 0
option.sitegroup[:] = 0
font = ImageFont.load_default(size=23)
fps = Fraction(1 / plan["control_dt"]).limit_denominator(1000)
args.output.parent.mkdir(parents=True, exist_ok=True)
frame_count = 0
maximum_pose_error = 0.
with h5py.File(args.trajectory, "r") as trace, mujoco.Renderer(model, height=720, width=1280) as renderer, av.open(str(args.output), "w") as video:
    stream = video.add_stream("libx264", rate=fps)
    stream.width, stream.height = 1280, 720
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}
    for index in range(len(trace["joint_position"])):
        # 视频恢复已经记录的姿态，物理验收使用原始轨迹报告。
        data.qpos[qids] = trace["joint_position"][index] + JOINT_OFFSETS
        data.qpos[oq:oq+3] = trace["object_position_root"][index]
        data.qpos[oq+3:oq+7] = trace["object_quaternion_root"][index]
        mujoco.mj_forward(model, data)
        error = float(np.linalg.norm(data.xpos[gid] - trace["gripper_position_root"][index]))
        if error > 1e-8:
            raise RuntimeError("视频机器人姿态与真实记录不一致")
        maximum_pose_error = max(maximum_pose_error, error)
        renderer.update_scene(data, camera=camera, scene_option=option)
        pixels = renderer.render()
        if pixels.shape != (720, 1280, 3) or pixels.dtype != np.uint8 or np.std(pixels) < 1:
            raise RuntimeError("MuJoCo 视频帧检查失败")
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1280, 90), fill=(14, 24, 38))
        height = 1000*(data.qpos[oq+2]-plan["object_position_root"][2])
        forces = trace["jaw_forces"][index]
        draw.text((22, 14), "OpenSO-101 | MuJoCo physical gripper verification", font=font, fill="white")
        draw.text((22, 49), f"{plan['phases'][index]} | t={(index+1)*plan['control_dt']:.2f}s | lift={height:.1f} mm | jaws={forces[0]:.2f}/{forces[1]:.2f} N", font=font, fill=(122, 210, 255))
        draw.rectangle((0, 676, 1280, 720), fill=(14, 24, 38))
        draw.text((22, 685), "Recorded physics trajectory | Scripted IK control | RL policy acceptance pending", font=font, fill="white")
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")):
            video.mux(packet)
        frame_count += 1
    for packet in stream.encode():
        video.mux(packet)
with av.open(str(args.output)) as video:
    decoded = sum(1 for frame in video.decode(video=0) if frame.width == 1280 and frame.height == 720)
if decoded != frame_count:
    raise RuntimeError("MP4 全部帧解码检查失败")
result = {"status": "recorded_gripper_video_verified", "frames": frame_count, "fps": float(fps),
          "duration_seconds": frame_count/float(fps), "size": [1280, 720],
          "maximum_robot_pose_error_m": maximum_pose_error, "trajectory_sha256": digest(args.trajectory),
          "video_sha256": digest(args.output), "rendering_source_sha256": digest(Path(__file__)),
          "rl_policy_success_verified": False}
args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
