import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openso101.scenes.models import file_digest

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
root = args.root.resolve()
result = json.loads((root / "prepared/agent_loop_result.json").read_text())
preparation = json.loads((root / "prepared/preparation.json").read_text())
spec = json.loads((root / "bundle/scene.json").read_text())
capture = json.loads((root / "capture/capture.json").read_text())
if capture["scene_sha256"] != preparation["scene_sha256"] or result["scene_sha256"] != preparation["scene_sha256"]:
    raise ValueError("演示源场景 SHA256 不一致")
if capture["frames"] != 720 or capture["fps"] != 60:
    raise ValueError("演示录制需要 720 帧和 60 FPS")
font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
fonts = {size: ImageFont.truetype(font_path, size) for size in (26, 32, 38, 48, 64)}
background = (9, 18, 32)
foreground = (228, 237, 248)
accent = (79, 185, 247)
muted = (153, 174, 198)
entity = spec["entities"][0]
asset = json.loads((root / "bundle/assets" / entity["asset_uid"] / "asset.json").read_text())
frame_paths = sorted((root / "input_frames").glob("*.jpg"))
if len(frame_paths) != 2:
    raise ValueError("演示需要实际模型输入的两个采样帧")
input_frame = Image.open(frame_paths[0]).convert("RGB")


def text(draw, value, x, y, size=38, color=foreground, width=1740):
    line = ""
    for char in value:
        candidate = line + char
        if draw.textlength(candidate, font=fonts[size]) > width:
            draw.text((x, y), line, font=fonts[size], fill=color)
            y += int(size * 1.5)
            line = char
        else:
            line = candidate
    draw.text((x, y), line, font=fonts[size], fill=color)
    return y + int(size * 1.5)


def place(canvas, picture, box):
    x, y, width, height = box
    image = picture.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))


def base(stage, title):
    canvas = Image.new("RGB", (1920, 1080), background)
    draw = ImageDraw.Draw(canvas)
    text(draw, "OpenSO-101 · SceneAgent", 80, 48, 32, accent)
    text(draw, title, 80, 124, 64)
    text(draw, f"{stage} / 5", 1740, 56, 32, muted)
    text(draw, "真实运行结果 · 仿真录制输入 · 任务成功待验收", 80, 1006, 26, muted)
    return canvas, draw


videos = {}
for name in ("overhead_camera", "wrist_camera"):
    path = root / "capture" / f"{name}.mp4"
    if file_digest(path) != capture["videos"][name]["sha256"]:
        raise ValueError(f"演示相机文件 SHA256 不一致: {name}")
    container = av.open(str(path))
    videos[name] = (container, iter(container.decode(video=0)))
first_frames = {name: next(frames).to_image().convert("RGB") for name, (_, frames) in videos.items()}
output = root / "OpenSO101-agentic-demo.mp4"
if output.exists():
    raise FileExistsError(output)
fps = 30
with av.open(str(output), "w") as movie:
    stream = movie.add_stream("libx264", rate=fps)
    stream.width, stream.height = 1920, 1080
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "19", "preset": "fast"}
    for index in range(35 * fps):
        seconds = index / fps
        if seconds < 6:
            canvas, draw = base(1, "描述任务，提供场景参考")
            text(draw, result["video"]["context"]["instruction"], 88, 270, 48, width=1030)
            text(draw, "视频采样：2 帧 · 128×128 · 60 FPS", 88, 650, 32, accent, width=1050)
            text(draw, "输入来源：已有双相机仿真录制", 88, 720, 32, muted, width=1050)
            place(canvas, input_frame.resize((640, 640)), (1180, 270, 640, 640))
        elif seconds < 11:
            canvas, draw = base(2, "Agent 检索与准备 3D 资产")
            y = text(draw, "模型根据任务和视频生成资产查询，读取已有资产目录。", 88, 270, 48, width=1680)
            y = text(draw, f"场景实体：{len(spec['entities'])} · 所用资产：{asset.get('name', entity['entity_id'])}", 88, y + 60, 48, accent)
            y = text(draw, f"尺寸：{entity['dimensions_m']} 米", 88, y + 24, 38)
            y = text(draw, f"本次生成资产：{len(result['generated_asset_uids'])} 项", 88, y + 24, 38, accent)
            text(draw, "资产来源、许可证、文件内容与 SHA256 保存在 bundle 中。", 88, y + 70, 38, muted)
        elif seconds < 17:
            canvas, draw = base(3, "Agent 编排任务场景")
            place(canvas, first_frames["overhead_camera"], (80, 270, 780, 680))
            y = text(draw, f"物体位置：{entity['pose']['position']} 米", 960, 290, 38, width=850)
            y = text(draw, f"任务目标：{spec['task']['goal_position_m']} 米", 960, y + 35, 38, width=850)
            y = text(draw, "配置检查、模型审查和 USD 编译使用同一场景版本。", 960, y + 80, 38, accent, width=850)
            text(draw, "画面来自该生成场景的实际 Isaac 相机。", 960, y + 65, 32, muted, width=850)
        elif seconds < 29:
            canvas, draw = base(4, "Isaac 中的实际双相机运行")
            if index == 17 * fps:
                frames_now = first_frames
            else:
                frames_now = {}
                for name, (_, frames) in videos.items():
                    next(frames)
                    frames_now[name] = next(frames).to_image().convert("RGB")
            place(canvas, frames_now["overhead_camera"], (80, 265, 840, 660))
            place(canvas, frames_now["wrist_camera"], (1000, 265, 840, 660))
            text(draw, "Overhead camera", 150, 927, 32, accent)
            text(draw, "Wrist camera", 1090, 927, 32, accent)
            text(draw, "机器人动作来自已保存的真实键盘采集记录", 960, 220, 26, muted, width=890)
        else:
            canvas, draw = base(5, "保存运行证据，保留验收状态")
            runtime = preparation["runtime"]
            y = text(draw, f"{runtime['num_envs']} 个环境 · {runtime['resets']} 次 reset · {runtime['steps']} 个控制步骤", 88, 286, 48, accent)
            y = text(draw, f"每个相机检查 {runtime['camera_checks']['wrist_camera']['checked_frames']} 帧", 88, y + 32, 48)
            y = text(draw, f"模型审查状态：{result['status']}", 88, y + 65, 38)
            y = text(draw, "程序运行检查：runtime_verified", 88, y + 24, 38)
            text(draw, "任务接触、物体可见性、成功采集与真机运行仍需对应验收。", 88, y + 75, 38, muted)
        frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(frame):
            movie.mux(packet)
    for packet in stream.encode():
        movie.mux(packet)
for container, _ in videos.values():
    container.close()
with av.open(str(output)) as container:
    video = container.streams.video[0]
    count = 0
    for frame in container.decode(video=0):
        pixels = frame.to_ndarray(format="rgb24")
        if pixels.shape != (1080, 1920, 3) or np.std(pixels) <= 1:
            raise ValueError("演示视频帧无效")
        count += 1
    if count != 1050 or float(video.average_rate) != 30:
        raise ValueError("演示视频帧数或 FPS 无效")
report = {"status": "demo_video_verified", "path": str(output), "sha256": file_digest(output),
          "frames": count, "fps": 30, "duration_seconds": 35, "resolution": [1920, 1080],
          "scene_sha256": capture["scene_sha256"], "task_success_verified": False,
          "agent_result_sha256": file_digest(root / "prepared/agent_loop_result.json"),
          "capture_report_sha256": file_digest(root / "capture/capture.json"),
          "builder_sha256": file_digest(Path(__file__))}
(root / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
