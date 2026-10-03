import argparse
from contextlib import ExitStack
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("scene", type=Path)
parser.add_argument("episode", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, default=720)
args = parser.parse_args()
if args.steps <= 0:
    raise ValueError("steps 必须为正数")

from openso101.rl.gpu_scope import configure_visible_gpu

configure_visible_gpu()
args.output.mkdir(parents=True, exist_ok=False)
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app
import av
import gymnasium as gym
import h5py
import numpy as np
import torch

from openso101.cli.il import _replay_set_robot_proprio
from openso101.scenes.models import file_digest
from openso101.scenes.runtime import CustomSceneEnvCfg, register_custom_scene
from openso101.scenes.usd import verify_compilation

compilation = verify_compilation(args.scene)
register_custom_scene()
cfg = CustomSceneEnvCfg()
cfg.configure_play(True)
cfg.configure_action_mode("teleop")
cfg.configure_scene(args.scene)
cfg.configure_cameras(True)
cfg.scene.num_envs = 1
for name in ("overhead_camera", "wrist_camera"):
    camera = getattr(cfg.scene, name)
    camera.width = 512
    camera.height = 512
env = gym.make("OpenSO101-CustomScene-v0", cfg=cfg)
report = {"scene_sha256": compilation["scene_sha256"], "motion_episode_sha256": file_digest(args.episode),
          "captured_env_index": 0, "frames": 0, "fps": 60, "resolution": [512, 512],
          "task_success_verified": False, "hardware_run_verified": False, "videos": {},
          "capture_source_sha256": file_digest(Path(__file__))}
try:
    env.reset()
    with ExitStack() as cleanup:
        episode = cleanup.enter_context(h5py.File(args.episode, "r"))
        if args.steps > len(episode["action"]) or not np.isclose(env.unwrapped.step_dt, 1 / report["fps"], atol=1e-6):
            raise ValueError("动作帧数或控制周期不匹配")
        _replay_set_robot_proprio(env.unwrapped.scene, episode["observations/qpos"][0], episode["observations/qvel"][0])
        streams = {}
        for name in ("overhead_camera", "wrist_camera"):
            path = args.output / f"{name}.mp4"
            container = cleanup.enter_context(av.open(str(path), "w"))
            stream = container.add_stream("libx264", rate=report["fps"])
            stream.width = stream.height = 512
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            streams[name] = (container, stream, path)
        for index in range(args.steps):
            action = torch.as_tensor(episode["action"][index], device=env.unwrapped.device).unsqueeze(0)
            observation, reward, _, _, _ = env.step(action)
            if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
                raise ValueError(f"录制步骤 {index} 产生无效状态")
            if torch.max(torch.abs(env.unwrapped.action_manager.action - action)) > 1e-6:
                raise ValueError(f"录制步骤 {index} 动作与源数据不一致")
            for name, (container, stream, _) in streams.items():
                rgb = env.unwrapped.scene[name].data.output["rgb"][0, :, :, :3].cpu().numpy()
                if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8 or np.std(rgb) <= 0:
                    raise ValueError(f"相机数据无效: {name}")
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            report["frames"] += 1
        for container, stream, _ in streams.values():
            for packet in stream.encode():
                container.mux(packet)
    for name, (_, _, path) in streams.items():
        report["videos"][name] = {"path": str(path.resolve()), "sha256": file_digest(path)}
    report["status"] = "scene_camera_capture_verified"
    (args.output / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
finally:
    env.close()
app.close()
