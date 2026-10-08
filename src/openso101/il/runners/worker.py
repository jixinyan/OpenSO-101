import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from openso101.rl.gpu_guard import verify_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("IL worker 缺少训练命令")
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    verify_guard(gpu)
    receipt = json.loads(Path(os.environ["OPENSO101_GPU_GUARD_REPORT"]).read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = receipt["initial_device"]["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("LeRobot 训练需要当前指定的单张 CUDA GPU")
    with args.log.open("x") as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
