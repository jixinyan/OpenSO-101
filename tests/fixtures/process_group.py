import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--child", action="store_true")
parser.add_argument("--parent-exit", action="store_true")
parser.add_argument("--pid-file", type=Path, required=True)
args = parser.parse_args()
if args.child:
    args.pid_file.write_text(str(os.getpid()))
else:
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", "--pid-file", str(args.pid_file)])
    if args.parent_exit:
        raise SystemExit(0)
time.sleep(120)
