import argparse
import json
import os
import subprocess
import shutil
import sys
import threading
from pathlib import Path

from openso101.il.datasets.export import _push_convert_hdf5_to_lerobot, _push_validate_local_dataset
from openso101.scenes.models import file_digest


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise ValueError("实际导出检查需要禁止 CUDA")
args.output.mkdir(parents=True, exist_ok=False)
sources = _push_validate_local_dataset(args.source, input_format="hdf5")
source_hashes = {str(path): file_digest(path) for path in sources}
results = {}
for mode, async_flush in (("async", True), ("sync", False)):
    root = args.output / mode
    _push_convert_hdf5_to_lerobot(args.source, root, f"local/export_{mode}",
                                skip_leading_frames=0, min_episode_frames=1, async_flush=async_flush)
    _push_validate_local_dataset(root, input_format="lerobot")
    subprocess.run([sys.executable, "scripts/validate_lerobot_dataset.py", "--source", str(args.source),
                    "--dataset", str(root), "--require-success", "--output", str(args.output / f"{mode}.json")],
                   check=True)
    results[mode] = json.loads((args.output / f"{mode}.json").read_text())
    artifacts = {path: file_digest(path) for path in root.rglob("*") if path.is_file()}
    try:
        _push_convert_hdf5_to_lerobot(args.source, root, f"local/export_{mode}",
                                    min_episode_frames=10**9, overwrite_export=True)
    except ValueError as exc:
        if "全部 episode" not in str(exc):
            raise
    else:
        raise RuntimeError("没有可导出 episode 时需要终止运行")
    if any(not path.is_file() or file_digest(path) != expected for path, expected in artifacts.items()):
        raise ValueError("输入检查终止后原有导出内容发生改变")
if any(thread.name.startswith("lerobot-flush") for thread in threading.enumerate()):
    raise RuntimeError("导出完成后仍有编码线程")
incomplete = args.output / "missing_video"
shutil.copytree(args.output / "sync", incomplete)
missing_video = next((incomplete / "videos").rglob("*.mp4"))
missing_video.unlink()
try:
    _push_validate_local_dataset(incomplete, input_format="lerobot")
except FileNotFoundError as exc:
    if str(missing_video.resolve()) not in str(exc):
        raise
else:
    raise RuntimeError("实际视频文件缺少时需要在数据读取前终止")
if any(file_digest(Path(path)) != expected for path, expected in source_hashes.items()):
    raise ValueError("实际来源数据发生改变")
for key in ("frames", "episodes", "source_task_successes", "maximum_action_error_motor_units",
            "maximum_state_error_motor_units", "maximum_timestamp_error_seconds", "camera_checks"):
    if results["async"][key] != results["sync"][key]:
        raise ValueError(f"同步与异步导出内容不一致: {key}")
result = {"status": "actual_sync_async_export_verified", "modes": results,
          "source_files": source_hashes, "worker_threads_closed": True, "gpu_tests_started": False,
          "invalid_export_preserves_existing_data": True,
          "rl_policy_success_verified": False,
          "source_sha256": file_digest(Path(__file__)),
          "exporter_sha256": file_digest(Path("src/openso101/il/datasets/export.py")),
          "metadata_validation_sha256": file_digest(Path("src/openso101/il/datasets/validation.py")),
          "metadata_frame_ranges_and_files_verified": True,
          "missing_actual_video_rejected": True}
(args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False), flush=True)
