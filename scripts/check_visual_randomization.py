import argparse
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.task = "OpenSO101-Lift-v0"
args.task_profile = "grasp_v4"
args.environment_mode = "nominal"
args.num_envs = 4
args.seed = 42
args.with_cameras = True
args.visual_dr = True

from openso101.rl.gpu_scope import configure_visible_gpu

configure_visible_gpu()
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app
env = None
try:
    import numpy as np
    import torch

    from openso101.rl.config import digest
    from openso101.rl.execution import build_environment
    from openso101.sim2real.domain_randomization.visual import _iter_preview_surface_attributes, _scene_asset_prim

    env = build_environment(args, training=True)
    runtime = env.unwrapped
    stage = runtime.sim.stage
    attributes = list(_iter_preview_surface_attributes(_scene_asset_prim(runtime, "object")))
    if not attributes:
        raise RuntimeError("实际物体缺少 color attribute")
    samples = []
    parameters = []
    for reset in range(3):
        env.reset()
        for _ in range(10):
            env.step(torch.zeros((4, 6), device=runtime.device))
        light = stage.GetPrimAtPath("/World/light")
        sample = {"reset": reset, "light_intensity": float(light.GetAttribute("inputs:intensity").Get()),
                  "light_color": list(light.GetAttribute("inputs:color").Get()),
                  "object_colors": [list(stage.GetPrimAtPath(path).GetAttribute(name).Get())
                                    for path, name in attributes], "cameras": {}}
        for name in ("overhead", "wrist"):
            camera = runtime.scene[f"{name}_camera"]
            rgb = camera.data.output["rgb"].cpu().numpy()
            if rgb.ndim != 4 or rgb.shape[0] != 4 or not np.isfinite(rgb).all() or rgb.max() == rgb.min():
                raise RuntimeError(f"{name} camera 的实际 RGB 无效")
            sample["cameras"][name] = {"shape": list(rgb.shape), "sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                                       "minimum": int(rgb.min()), "maximum": int(rgb.max())}
        robot = runtime.scene["robot"]
        parameters.append({"mass": robot.root_physx_view.get_masses().cpu().numpy().copy(),
                           "stiffness": robot.root_physx_view.get_dof_stiffnesses().cpu().numpy().copy()})
        samples.append(sample)
    for name in ("light_intensity", "light_color", "object_colors"):
        if all(samples[0][name] == item[name] for item in samples[1:]):
            raise RuntimeError(f"实际 visual DR 数值没有变化：{name}")
    for name in parameters[0]:
        if any(not np.array_equal(parameters[0][name], item[name]) for item in parameters[1:]):
            raise RuntimeError(f"nominal physics 参数发生变化：{name}")
    report = {"status": "native_visual_randomization_verified", "task": args.task, "task_profile": args.task_profile,
              "environment_mode": args.environment_mode, "visual_dr": True, "environments": 4,
              "resets": 3, "control_steps_per_reset": 10, "samples": samples,
              "nominal_mass_and_stiffness_unchanged": True,
              "source_sha256": digest(Path(__file__)), "task_success_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
finally:
    if env is not None:
        env.close()
app.close()
