import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np

from openso101.rl.config import digest
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode


def main():
    parser = argparse.ArgumentParser(description="编码并验证真实策略双相机 episode")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    validate_hdf5_episode(args.episode)
    if digest(args.episode) != evaluation["recorded_episode"]["sha256"]:
        raise ValueError("episode SHA256 与评估记录不一致")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.episode, "r") as episode:
        policy_sha256 = str(episode.attrs["policy_sha256"])
        if policy_sha256 != evaluation.get("student_sha256", evaluation["checkpoint_sha256"]):
            raise ValueError("录制策略与评估模型的 SHA256 不一致")
        records = [item for item in evaluation["episodes"] if item["env_index"] == 0]
        record = records[0]
        frames = len(episode["action"])
        if frames != record["steps"] or bool(episode.attrs["success"]) != record["success"]:
            raise ValueError("首个环境的完整步骤数量或成功标记与评估不一致")
        fps = Fraction(str(episode.attrs["fps"]))
        timestamps = episode["timestamps"][:]
        if not np.allclose(np.diff(timestamps), float(1 / fps), atol=1e-9, rtol=0):
            raise ValueError("帧时间戳与控制频率不一致")
        overhead = episode["observations/images/overhead_camera"]
        wrist = episode["observations/images/wrist_camera"]
        if overhead.shape != wrist.shape:
            raise ValueError("两个相机尺寸不一致")
        height, width = overhead.shape[1:3]
        if width % 2 or height % 2:
            raise ValueError("H.264 视频尺寸需要偶数")
        with av.open(str(args.output), "w") as movie:
            stream = movie.add_stream("libx264", rate=fps)
            stream.width, stream.height = width * 2, height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            for index in range(frames):
                views = [overhead[index], wrist[index]]
                if any(np.std(view) == 0 for view in views):
                    raise ValueError(f"第 {index} 帧存在空白相机")
                image = np.concatenate(views, axis=1)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    movie.mux(packet)
            for packet in stream.encode():
                movie.mux(packet)
    decoded = 0
    with av.open(str(args.output)) as movie:
        if movie.streams.video[0].average_rate != fps:
            raise ValueError("视频帧率与控制频率不一致")
        for frame in movie.decode(video=0):
            if (frame.width, frame.height) != (width * 2, height):
                raise ValueError("解码视频尺寸不一致")
            decoded += 1
    if decoded != frames:
        raise ValueError("视频解码帧数与完整 episode 不一致")
    report = {
        "video_sha256": digest(args.output), "episode_sha256": digest(args.episode),
        "evaluation_sha256": digest(args.evaluation), "policy_sha256": policy_sha256,
        "frames": frames, "fps": float(fps), "duration_seconds": frames / float(fps),
        "width": width * 2, "height": height, "views": ["overhead_camera", "wrist_camera"],
        "seed": evaluation["seed"], "task": evaluation["task"],
        "completed_transitions": evaluation["completed_transitions"],
        "training_git_sha": evaluation["training_git_sha"],
        "evaluation_git_sha": evaluation["evaluation_git_sha"], "episode": record,
        "evaluation_success_rate": evaluation["success_rate"], "decoded_frames": decoded,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
