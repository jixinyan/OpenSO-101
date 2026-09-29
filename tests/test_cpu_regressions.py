"""CPU-level regression checks for the data and scene boundaries.

These tests deliberately avoid importing Isaac Lab so they can run in a
regular Python 3.11 environment.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from pydantic import BaseModel

from openso101.il.datasets.lerobot_adapter import LeRobotDatasetHandle
from openso101.scenes import model_client
from openso101.scenes.agent_loop import (
    DraftEntity,
    PlausibilityReview,
    RGBVideoInput,
    Real2SimAgentLoop,
    SceneDraft,
    SO101ReadinessReview,
    TrimeshAssetGenerator,
    VideoObject,
    VideoSceneDescription,
)
from openso101.scenes.catalog import AssetCatalog
from openso101.scenes.bundle import verify_bundle
from openso101.scenes.models import Asset
from openso101.scenes.video import sample_rgb_video
from openso101.teleop.hdf5_recorder import (
    OpenSO101HDF5TeleopRecorder,
    validate_hdf5_episode,
)
from openso101.teleop.so101_mapping import (
    batched_action_to_motor_units,
    batched_motor_units_to_action,
    transform_ordered_targets,
)


def test_dataset_handle_does_not_eagerly_call_len():
    class Dataset:
        num_frames = 12
        num_episodes = 2

    handle = LeRobotDatasetHandle(Dataset(), "local/demo", None)
    assert handle.num_frames == 12
    assert handle.num_episodes == 2


def test_hdf5_recorder_rejects_invalid_frame_and_round_trips(tmp_path):
    cameras = {name: {"height": 2, "width": 3} for name in ("wrist_camera", "overhead_camera")}
    recorder = OpenSO101HDF5TeleopRecorder(tmp_path, "demo", cameras, fps=30, flush_steps=1)
    recorder.start_episode()
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="action"):
        recorder.add_frame(
            np.zeros(5), np.zeros(6), np.zeros(6),
            camera_buffers={k: image for k in cameras}, timestamp=0.0,
        )
    recorder.add_frame(
        np.zeros(6), np.zeros(6), np.zeros(6),
        {name: image for name in cameras}, timestamp=0.0,
    )
    episode = recorder.save_episode(success=True)
    assert episode is not None
    validate_hdf5_episode(episode)


def test_mapping_round_trip_and_unknown_joint_guard():
    values = torch.zeros(6)
    motors = batched_action_to_motor_units(values)
    recovered = batched_motor_units_to_action(motors)
    assert torch.allclose(values, recovered, atol=1e-5)
    with pytest.raises(ValueError, match="Unknown"):
        transform_ordered_targets(values.tolist(), inverted_joints=("unknown",))


def test_asset_bounds_must_have_positive_extent():
    with pytest.raises(ValueError, match="bounds"):
        Asset(
            uid="a" * 32,
            name="test",
            source_url="local:test",
            license="MIT",
            author="test",
            sha256="b" * 64,
            bounds=((0.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
            vertices=1,
            faces=1,
        )


def test_model_service_accepts_no_auth_and_object_json(monkeypatch, tmp_path):
    class Output(BaseModel):
        value: int

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": {"value": 3}}}]}

    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, endpoint, *, headers, json):
            assert endpoint.endswith("/chat/completions")
            assert headers == {}
            assert isinstance(json["messages"][1]["content"], list)
            assert json["messages"][1]["content"][1]["type"] == "image_url"
            return Response()

    monkeypatch.delenv("OPEN_SO_TEST_KEY", raising=False)
    monkeypatch.setattr(model_client.httpx, "Client", Client)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not-a-real-jpeg")
    result = model_client.ModelService("http://localhost/v1", "test", "OPEN_SO_TEST_KEY").complete(
        system="system", prompt="prompt", schema=Output, images=(frame,)
    )
    assert result.value == 3


def test_model_service_responses_wire_and_codex_config(tmp_path, monkeypatch):
    class Output(BaseModel):
        value: int

    class Response:
        headers = {"content-type": "text/event-stream"}

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield 'data: {"type":"response.output_text.delta","delta":"{\\"value\\": 4}"}'
            yield 'data: {"type":"response.completed","response":{"status":"completed"}}'

    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, endpoint, *, headers, json):
            assert endpoint.endswith("/responses")
            assert headers == {"Authorization": "Bearer test-token"}
            assert json["input"][0]["role"] == "developer"
            assert json["input"][0]["content"][0]["text"].startswith("system")
            assert json["input"][1]["content"][0]["text"] == "prompt"
            assert json["text"] == {"verbosity": "low"}
            assert json["reasoning"] == {"effort": "xhigh", "summary": "auto"}
            assert json["stream"] is True
            assert json["store"] is False
            return Response()

        @contextmanager
        def stream(self, method, endpoint, *, headers, json):
            assert method == "POST"
            yield self.post(endpoint, headers=headers, json=json)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'model = "gpt-6-astra"\n'
        'model_provider = "litchi"\n'
        'model_reasoning_effort = "xhigh"\n'
        'model_reasoning_summary = "auto"\n'
        '\n'
        '[model_providers.litchi]\n'
        'name = "litchi"\n'
        'base_url = "https://example.invalid/v1"\n'
        'wire_api = "responses"\n'
        'experimental_bearer_token = "test-token"\n'
        'model = "ignored-provider-model"\n',
    )
    config = model_client.load_codex_runtime_config(config_path)
    assert config.model == "gpt-6-astra"
    assert config.base_url == "https://example.invalid/v1"
    assert config.wire_api == "responses"
    assert config.reasoning_effort == "xhigh"
    assert "test-token" not in repr(config)

    monkeypatch.setenv("OPEN_SO_RESPONSES_KEY", config.bearer_token or "")
    monkeypatch.setattr(model_client.httpx, "Client", Client)
    result = model_client.ModelService(
        config.base_url, config.model, "OPEN_SO_RESPONSES_KEY", wire_api=config.wire_api,
    ).complete(system="system", prompt="prompt", schema=Output)
    assert result.value == 4


def test_real2sim_agent_loop_materializes_generated_asset_and_bundle(tmp_path):
    class Planner:
        def describe(self, video):
            return VideoSceneDescription(
                instruction="将方块放到目标位置并释放夹爪。",
                task_family="pick_place",
                objects=(VideoObject(entity_id="object", label="block", asset_query="block", confidence=0.9),),
            )

        def compose(self, video, description, candidates):
            return SceneDraft(
                scene_id="agent_scene",
                entities=(DraftEntity(
                    entity_id="object", asset_query="block", primitive="box",
                    dimensions_m=(0.06, 0.06, 0.04),
                    pose={"position": (0.25, 0.0, 0.02)},
                ),),
                task={
                    "task_id": "pick_place",
                    "object_id": "object",
                    "goal_position_m": (0.25, 0.1, 0.02),
                    "instruction": "将方块放到目标位置并释放夹爪。",
                },
            )

        def review_physical(self, video, spec, static_diagnostics):
            return PlausibilityReview(approved=True, confidence=0.9)

        def review_so101(self, video, spec, physical):
            return SO101ReadinessReview(
                approved=True, confidence=0.8, reachable=True, camera_visible=True, task_ready=True,
            )

    class EmptyRetriever:
        def search(self, request):
            return ()

        def materialize(self, candidate):
            raise AssertionError("generated path should not materialize a candidate")

    result = Real2SimAgentLoop(
        catalog=AssetCatalog(tmp_path / "assets"),
        planner=Planner(),
        retriever=EmptyRetriever(),
        generator=TrimeshAssetGenerator(),
    ).run(RGBVideoInput(source="capture.mp4", frame_count=1, fps=30, width=8, height=8), output=tmp_path / "bundle")
    assert result.status == "completed"
    assert len(result.generated_asset_uids) == 1
    assert result.bundle is not None
    assert (tmp_path / "bundle" / "manifest.json").is_file()
    assert verify_bundle(tmp_path / "bundle").scene_id == "agent_scene"


def test_rgb_video_sampling_is_deterministic_and_keeps_timestamps(tmp_path):
    import av

    source = tmp_path / "capture.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height = 16, 12
        stream.pix_fmt = "yuv420p"
        for index in range(5):
            array = np.full((12, 16, 3), index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = sample_rgb_video(source, tmp_path / "frames", count=3, max_edge=64)
    assert video.frame_count == 5
    assert len(video.frame_paths) == len(video.timestamps_seconds) == 3
    assert video.width == 16 and video.height == 12
    assert all((tmp_path / "frames" / f"frame_{index:08d}.jpg").is_file() for index in (0, 2, 4))
