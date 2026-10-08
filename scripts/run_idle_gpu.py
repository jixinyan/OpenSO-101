import argparse
import os
import sys
from pathlib import Path

from openso101.rl.gpu_guard import automatic_report, run_guarded


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--report", type=Path)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    raise ValueError("GPU 启动需要明确的程序入口")
repo = Path(os.environ["OPENSO101_REPO"])
sys.exit(run_guarded(command, args.gpu, repo, args.report or automatic_report(repo)))
