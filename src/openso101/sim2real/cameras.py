import json
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np

from openso101.rl.config import digest

from .deploy import _camera_frame_tensor, _camera_source, _open_cameras, _validate_camera_arguments


def check_cameras(args) -> int:
    _validate_camera_arguments(args)
    if args.frames <= 0:
        raise ValueError("frames 必须为正数")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"相机报告已存在: {output}")
    sources = {f"{name}_camera": _camera_source(args, name) for name in ("wrist", "overhead")}
    report = {
        "status": "camera_read_running",
        "requested_frames": args.frames,
        "completed_frames": 0,
        "requested_fps": args.fps,
        "image_tensor_shape": [1, 3, args.camera_height, args.camera_width],
        "hardware_run_verified": False,
        "cameras": {},
        "checker_sha256": digest(Path(__file__)),
        "deploy_sha256": digest(Path(__file__).with_name("deploy.py")),
    }
    for name, source in sources.items():
        video_file = isinstance(source, Path) and source.is_file()
        report["cameras"][name] = {
            "source": str(source),
            "source_kind": "video_file" if video_file else "device",
            "source_sha256": digest(source) if video_file else None,
            "frames": 0,
            "minimum_pixel_std": None,
        }
    cameras = _open_cameras(
        wrist_index=sources["wrist_camera"], overhead_index=sources["overhead_camera"],
        width=args.camera_width, height=args.camera_height, fps=args.fps,
    )
    start = time.perf_counter()
    with ExitStack() as cleanup:
        for camera in cameras.values():
            cleanup.callback(camera.disconnect)
        for _ in range(args.frames):
            for name, camera in cameras.items():
                frame = camera.read()
                tensor = _camera_frame_tensor(frame)
                if list(tensor.shape) != report["image_tensor_shape"]:
                    raise ValueError(f"相机图像尺寸不匹配: {name}")
                item = report["cameras"][name]
                item["frames"] += 1
                pixel_std = float(np.std(frame))
                previous = item["minimum_pixel_std"]
                item["minimum_pixel_std"] = pixel_std if previous is None else min(previous, pixel_std)
            report["completed_frames"] += 1
        report["read_seconds"] = time.perf_counter() - start
        report["status"] = "camera_read_verified"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0
