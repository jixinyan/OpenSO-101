import argparse
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
args.with_cameras = False
args.visual_dr = False
args.output.mkdir(parents=True, exist_ok=False)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
env = None
try:
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from openso101.rl.backends.rsl_rl import configuration
    from openso101.rl.config import TrainCfg, digest
    from openso101.rl.execution import build_environment
    from openso101.rl.runners.on_policy_runner import BestCheckpointRunner

    env = build_environment(args, training=True)
    config = configuration(TrainCfg(iterations=4, rollout_steps=96), env.unwrapped.device)
    config["save_interval"] = 1
    runner = BestCheckpointRunner(RslRlVecEnvWrapper(env), config, log_dir=str(args.output), device=env.unwrapped.device)
    runner.learn(num_learning_iterations=4, init_at_random_ep_len=False)
    models = {}
    for name in ("model_best.pt", "model_best_reward.pt", "model_3.pt"):
        path = args.output / name
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not all(torch.isfinite(value).all() for value in saved["model_state_dict"].values()):
            raise RuntimeError("实际 runner 模型包含无效参数")
        models[name] = {"sha256": digest(path), "iteration": saved["iter"]}
    report = {"status": "native_checkpoint_selection_verified", "task": args.task, "task_profile": args.task_profile,
              "num_envs": 4, "iterations": 4, "transitions": 1536, "models": models,
              "training_log_success": runner._best_success, "source_sha256": digest(Path(__file__)),
              "runner_source_sha256": digest(Path("src/openso101/rl/runners/on_policy_runner.py")),
              "independent_task_success_verified": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
finally:
    if env is not None:
        env.close()
app.close()
