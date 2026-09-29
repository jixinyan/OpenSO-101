# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""RGB video decoding and explicit metric context for reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import Field, model_validator

from .models import Digest, Dimensions, Identifier, Model, Pose, Table, file_digest


class SceneContext(Model):
    instruction: str = ""
    table: Table | None = None
    robot_base: Pose | None = None
    object_dimensions_m: dict[Identifier, Dimensions] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


class RGBVideoInput(Model):
    source: str = Field(min_length=1)
    source_sha256: Digest | None = None
    frame_count: int = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_paths: tuple[str, ...] = ()
    timestamps_seconds: tuple[float, ...] = ()
    context: SceneContext = SceneContext()

    @model_validator(mode="after")
    def valid_frames(self):
        if len(self.frame_paths) > self.frame_count:
            raise ValueError("frame_paths 不能多于 frame_count")
        times = self.timestamps_seconds
        if times and (len(times) != len(self.frame_paths) or any(t < 0 for t in times)
                      or any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError("timestamps_seconds 必须和采样帧一一对应且非负、单调")
        return self

    def model_images(self, limit: int = 8) -> tuple[str, ...]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 32:
            raise ValueError("limit 必须位于 1 到 32")
        indices = np.linspace(0, len(self.frame_paths) - 1, min(limit, len(self.frame_paths)), dtype=int)
        return tuple(self.frame_paths[index] for index in indices)


def sample_rgb_video(
    source: Path,
    output: Path,
    *,
    count: int = 8,
    max_edge: int = 1024,
    context: SceneContext | None = None,
) -> RGBVideoInput:
    """Decode two streaming passes so sampling works with missing frame metadata.

    The first pass counts actual decoded frames; the second saves evenly spaced
    frames, including the beginning and end. Memory use is one decoded frame.
    PyAV ships the codec libraries and does not need an ffmpeg executable.
    """
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 32:
        raise ValueError("count 必须位于 1 到 32")
    if not isinstance(max_edge, int) or isinstance(max_edge, bool) or max_edge < 64:
        raise ValueError("max_edge 必须至少为 64")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    try:
        import av
    except ImportError as exc:
        raise ImportError("视频抽帧需要 PyAV：安装 openso-101[scenes]") from exc
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError("输入文件没有视频轨道")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        frame_count = sum(1 for _ in container.decode(video=0))
    if frame_count == 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("视频没有可解码帧或有效帧率")
    wanted = set(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int).tolist())
    output.mkdir(parents=True, exist_ok=False)
    paths, times = [], []
    width = height = 0
    with av.open(str(source)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in wanted:
                continue
            image = frame.to_image().convert("RGB")
            width, height = image.size
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            path = output / f"frame_{index:08d}.jpg"
            image.save(path, quality=90)
            paths.append(str(path.resolve()))
            times.append(float(frame.time) if frame.time is not None else index / fps)
    if len(paths) != len(wanted):
        raise ValueError("视频第二次解码结果与帧计数不一致")
    # Normalize PTS to the video start while retaining actual sample intervals.
    start = times[0]
    return RGBVideoInput(
        source=str(source), source_sha256=file_digest(source), frame_count=frame_count,
        fps=fps, width=width, height=height, frame_paths=tuple(paths),
        timestamps_seconds=tuple(t - start for t in times), context=context or SceneContext(),
    )
