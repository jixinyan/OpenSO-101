import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

from openso101.rl.config import digest
from openso101.rl.plotting import load_scalar_data, smooth_values


parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--font", required=True, type=Path)
args = parser.parse_args()
if os.environ["CUDA_VISIBLE_DEVICES"] != "":
    raise ValueError("保存日志检查需要关闭 GPU")
files = sorted(args.run.glob("events.out.tfevents.*"))
if not files:
    raise ValueError("来源模型没有实际 event 文件")
original = {str(path): digest(path) for path in files}
args.output.mkdir(parents=True, exist_ok=False)
source = args.output / "events"
source.mkdir()
for path in files:
    shutil.copy2(path, source / path.name)
data, report = load_scalar_data(source)
accumulator = event_accumulator.EventAccumulator(
    str(source), size_guidance={event_accumulator.SCALARS: 0}, purge_orphaned_data=False,
).Reload()
if set(data) != set(accumulator.Tags()["scalars"]):
    raise ValueError("逐文件读取与 TensorBoard 目录读取的指标不相同")
for tag, (steps, values) in data.items():
    expected = {}
    for event in accumulator.Scalars(tag):
        if event.step not in expected or event.wall_time >= expected[event.step].wall_time:
            expected[event.step] = event
    keys = sorted(expected)
    if not np.array_equal(steps, keys) or not np.array_equal(values, [expected[key].value for key in keys]):
        raise ValueError(f"实际训练日志读取数值不同：{tag}")
    for size in (1, 2, 30, 10000):
        smoothed = smooth_values(values, size)
        if smoothed.shape != values.shape or not np.isfinite(smoothed).all():
            raise ValueError("训练曲线数值或 step 数量无效")
figures = args.output / "figures"
command = [sys.executable, "-m", "openso101.cli.main", "rl", "plot", "--run", str(source),
           "--output", str(figures), "--font-file", str(args.font)]
subprocess.run(command, check=True)
image_path = figures / "training_curves.png"
with Image.open(image_path) as image:
    image.verify()
with Image.open(image_path) as image:
    dimensions = [image.width, image.height]
metadata = json.loads((figures / "training_curves.json").read_text())
if metadata["files"] != report["files"] or metadata["image_sha256"] != digest(image_path):
    raise ValueError("实际图表与文件来源不同")
for path in files:
    if digest(path) != original[str(path)]:
        raise ValueError("来源日志发生改变")
with (args.output / "overwrite.log").open("x") as stream:
    repeated = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
if repeated.returncode == 0 or digest(image_path) != metadata["image_sha256"]:
    raise ValueError("图表需要保持已有记录")
result = {"status": "actual_event_data_and_cli_plot_verified", "event_files": len(files),
          "scalar_events": report["scalar_events"], "series": len(data), "dimensions": dimensions,
          "source_files": original, "image_sha256": digest(image_path),
          "source_sha256": digest(Path("src/openso101/rl/plotting.py")),
          "gpu_tests_started": False, "independent_task_success_verified": False}
(args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
