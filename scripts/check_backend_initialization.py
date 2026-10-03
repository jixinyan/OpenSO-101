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
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
config = TrainCfg(backend=args.backend, algo=args.algo, iterations=2, rollout_steps=32,
                  initial_noise_std=.73, environment_mode="nominal", learning_starts=16,
                  replay_size=512, replay_batch_size=16, hidden_dims=(32, 16))
config_path = args.output / "requested_config.json"
config_path.write_text(config.model_dump_json(indent=2))
run = args.output / "run"
command = [sys.executable, "-u", "-m", "openso101.cli.main", "rl", "train",
           "--task", "OpenSO101-Lift-v0", "--train-config", str(config_path),
           "--backend", args.backend, "--algo", args.algo, "--num_envs", "4",
           "--output", str(run), "--headless", "--no-video", "--logger", "tensorboard"]
with (args.output / "training.log").open("x") as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
metadata = CheckpointMeta.read(run)
initialization = json.loads((run / "policy_initialization.json").read_text())
if initialization["resumed"] or initialization["requested_initial_std"] != .73:
    raise ValueError("原生训练检查的策略初始化记录不一致")
command = [sys.executable, "-u", "-m", "openso101.cli.main", "rl", "eval",
           "--task", "OpenSO101-Lift-v0", "--checkpoint", str(run),
           "--num-envs", "4", "--n-episodes", "4", "--seed", "10042", "--headless"]
with (args.output / "evaluation.log").open("x") as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
evaluations = sorted(run.glob("evaluation-*.json"))
if len(evaluations) != 1:
    raise ValueError("原生 backend 检查需要一份独立评估")
evaluation = json.loads(evaluations[0].read_text())
if len(evaluation["episodes"]) != 4:
    raise ValueError("独立评估回合数量不一致")
report = {"status": "native_backend_initialization_verified", "backend": args.backend, "algo": args.algo,
          "training_transitions": metadata.completed_transitions, "initialization": initialization,
          "checkpoint_sha256": metadata.files[metadata.checkpoint], "source_sha256": digest(Path(__file__)),
          "evaluation_sha256": digest(evaluations[0]), "evaluation_success_rate": evaluation["success_rate"],
          "task_success_threshold_verified": False,
          "independent_evaluation_episodes": 4}
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
