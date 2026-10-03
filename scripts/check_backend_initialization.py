import argparse
import json
from pathlib import Path
import subprocess
import sys

from openso101.rl.config import CheckpointMeta, TrainCfg, digest


parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("rsl_rl", "sb3", "skrl", "rl_games"), required=True)
parser.add_argument("--algo", choices=("ppo", "sac", "tqc"), default="ppo")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task-profile", choices=("default", "grasp_v3", "grasp_v4"), default="default")
parser.add_argument("--iterations", type=int, default=2)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=4)
args = parser.parse_args()
if args.num_envs <= 0 or args.episodes <= 0:
    raise ValueError("backend 检查需要有效的环境与回合数量")
args.output.mkdir(parents=True, exist_ok=False)
config = TrainCfg(backend=args.backend, algo=args.algo, iterations=args.iterations, rollout_steps=32,
                  initial_noise_std=.73, environment_mode="nominal", learning_starts=16,
                  replay_size=512, replay_batch_size=16, hidden_dims=(32, 16),
                  action_distribution="tanh_gaussian" if args.backend == "rsl_rl"
                  and args.task_profile in ("grasp_v3", "grasp_v4") else "gaussian")
config.batch_size(args.num_envs)
config_path = args.output / "requested_config.json"
config_path.write_text(config.model_dump_json(indent=2))
run = args.output / "run"
command = [sys.executable, "-u", "-m", "openso101.cli.main", "rl", "train",
           "--task", "OpenSO101-Lift-v0", "--train-config", str(config_path),
           "--backend", args.backend, "--algo", args.algo, "--num_envs", str(args.num_envs),
           "--task-profile", args.task_profile,
           "--output", str(run), "--headless", "--no-video", "--logger", "tensorboard"]
with (args.output / "training.log").open("x") as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
metadata = CheckpointMeta.read(run)
initialization = json.loads((run / "policy_initialization.json").read_text())
if initialization["resumed"] or initialization["requested_initial_std"] != .73:
    raise ValueError("原生训练检查的策略初始化记录不一致")
command = [sys.executable, "-u", "-m", "openso101.cli.main", "rl", "eval",
           "--task", "OpenSO101-Lift-v0", "--checkpoint", str(run),
           "--num-envs", str(args.num_envs), "--n-episodes", str(args.episodes), "--seed", "10042", "--headless"]
with (args.output / "evaluation.log").open("x") as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
evaluations = sorted(run.glob("evaluation-*.json"))
if len(evaluations) != 1:
    raise ValueError("原生 backend 检查需要一份独立评估")
evaluation = json.loads(evaluations[0].read_text())
if len(evaluation["episodes"]) != args.episodes:
    raise ValueError("独立评估回合数量不一致")
report = {"status": "native_backend_initialization_verified", "backend": args.backend, "algo": args.algo,
          "task_profile": metadata.task_profile, "iterations": args.iterations, "num_envs": args.num_envs,
          "action_distribution": config.action_distribution,
          "training_transitions": metadata.completed_transitions, "initialization": initialization,
          "checkpoint_sha256": metadata.files[metadata.checkpoint], "source_sha256": digest(Path(__file__)),
          "evaluation_sha256": digest(evaluations[0]), "evaluation_success_rate": evaluation["success_rate"],
          "task_success_threshold_verified": False,
          "independent_evaluation_episodes": args.episodes}
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
