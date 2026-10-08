import argparse
import json
from pathlib import Path

import h5py

from openso101.il.runtime import _launch_isaac_app, _with_cleanup
from openso101.scenes.isaaclab.usd import verify_compilation
from openso101.scenes.models import file_digest
from openso101.teleop.recorder.hdf5 import validate_hdf5_episode
from openso101.teleop.sim_state import _replay_set_robot_proprio


@_with_cleanup
def capture(args, cleanup):
    import av
    import numpy as np

    if args.steps <= 0:
        raise ValueError("steps 必须为正数")
    if args.output.exists():
        raise FileExistsError(args.output)
    compilation = verify_compilation(args.scene)
    validate_hdf5_episode(args.episode)
    episode = cleanup.enter_context(h5py.File(args.episode, "r"))
    fps = int(episode.attrs["fps"])
    if args.steps > len(episode["action"]) or fps != 60:
        raise ValueError("动作帧数或场景录制的 60 Hz 控制频率不匹配")
    _launch_isaac_app(args, cleanup, enable_cameras=True)
    import gymnasium as gym
    import torch
    from openso101.scenes.isaaclab.runtime import CustomSceneEnvCfg

    cfg = CustomSceneEnvCfg()
    cfg.configure_play(True)
    cfg.configure_action_mode("teleop")
    cfg.configure_scene(args.scene)
    cfg.configure_cameras(True)
    cfg.scene.num_envs = 1
    for name in ("overhead_camera", "wrist_camera"):
        camera = getattr(cfg.scene, name)
        camera.width = camera.height = 512
    env = gym.make("OpenSO101-CustomScene-v0", cfg=cfg)
    cleanup.callback(env.close)
    env.reset()
    if not np.isclose(env.unwrapped.step_dt, 1 / fps, atol=1e-6):
        raise ValueError("场景控制周期与来源 FPS 不匹配")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scene_sha256": compilation["scene_sha256"], "motion_episode_sha256": file_digest(args.episode),
              "captured_env_index": 0, "frames": 0, "fps": fps, "resolution": [512, 512],
              "task_success_verified": False, "hardware_run_verified": False, "videos": {},
              "capture_source_sha256": file_digest(Path(__file__)),
              "sim_state_sha256": file_digest(Path("src/openso101/teleop/sim_state.py")),
              "runtime_sha256": file_digest(Path("src/openso101/il/runtime.py"))}
    _replay_set_robot_proprio(env.unwrapped.scene, episode["observations/qpos"][0], episode["observations/qvel"][0])
    streams = {}
    for name in ("overhead_camera", "wrist_camera"):
        path = args.output / f"{name}.mp4"
        container = cleanup.enter_context(av.open(str(path), "w"))
        stream = container.add_stream("libx264", rate=fps)
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
    # 编码完成并关闭文件后保存视频 SHA256。
    for container, stream, _ in streams.values():
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    for name, (_, _, path) in streams.items():
        report["videos"][name] = {"path": str(path.resolve()), "sha256": file_digest(path)}
    report["status"] = "scene_camera_capture_verified"
    (args.output / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


parser = argparse.ArgumentParser()
parser.add_argument("scene", type=Path)
parser.add_argument("episode", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, default=720)
args = parser.parse_args()
args.headless = True
raise SystemExit(capture(args))
