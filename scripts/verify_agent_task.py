import argparse
import json
from pathlib import Path

from openso101.scenes.agent.loop import AgentLoopResult
from openso101.scenes.bundle import verify_bundle
from openso101.scenes.models import file_digest
from openso101.scenes.isaaclab.usd import verify_compilation

parser = argparse.ArgumentParser()
parser.add_argument("prepared", type=Path)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--min-revisions", type=int, default=2)
args = parser.parse_args()
if args.min_revisions < 1:
    raise ValueError("min-revisions 必须为正数")
if args.report.exists():
    raise FileExistsError(args.report)
result_path = args.prepared / "agent_loop_result.json"
result = AgentLoopResult.model_validate_json(result_path.read_text())
instruction = result.video.context.instruction
if not instruction or result.description.instruction != instruction:
    raise ValueError("视频描述需要保留用户任务原文")
if len(result.scene_revisions) < args.min_revisions:
    raise ValueError("实际场景版本数量不足")
for index, revision in enumerate(result.scene_revisions):
    if revision.revision != index or revision.task_instruction != instruction or revision.instruction_source != "user_context":
        raise ValueError(f"场景版本 {index} 的任务原文或来源不一致")
spec = verify_bundle(Path(result.bundle))
compilation = verify_compilation(Path(result.compiled_scene))
preparation_path = args.prepared / "preparation.json"
preparation = json.loads(preparation_path.read_text())
runtime_path = args.prepared / "runtime.json"
runtime = json.loads(runtime_path.read_text())
if spec.task.instruction != instruction:
    raise ValueError("最终 bundle 需要保留用户任务原文")
if any(digest != spec.digest() for digest in (
    result.scene_sha256, result.scene_revisions[-1].scene_sha256,
    compilation["scene_sha256"], preparation["scene_sha256"], runtime["scene_sha256"],
)):
    raise ValueError("场景版本、bundle、编译与运行报告 SHA256 不一致")
if preparation["runtime_report_sha256"] != file_digest(runtime_path) or preparation["runtime"] != runtime:
    raise ValueError("运行报告内容与场景准备报告不一致")
if result.runtime_validation != runtime or runtime["status"] != "runtime_verified":
    raise ValueError("agent loop 与实际运行报告不一致")
expected_frames = runtime["num_envs"] * runtime["steps"]
for name in ("wrist_camera", "overhead_camera"):
    if runtime["camera_checks"][name]["checked_frames"] != expected_frames:
        raise ValueError(f"相机检查帧数不一致: {name}")
source = Path(result.video.source)
if file_digest(source) != result.video.source_sha256:
    raise ValueError("输入视频 SHA256 不一致")
report = {
    "status": "task_instruction_verified",
    "instruction": instruction,
    "scene_sha256": spec.digest(),
    "scene_revisions": [revision.model_dump() for revision in result.scene_revisions],
    "model_status": result.status,
    "video_sha256": result.video.source_sha256,
    "agent_result_sha256": file_digest(result_path),
    "preparation_sha256": file_digest(preparation_path),
    "runtime_report_sha256": file_digest(runtime_path),
    "verifier_sha256": file_digest(Path(__file__)),
    "agent_loop_sha256": file_digest(Path(__file__).resolve().parents[1] / "src/openso101/scenes/agent/loop.py"),
    "runtime": {
        "num_envs": runtime["num_envs"], "resets": runtime["resets"],
        "steps": runtime["steps"], "camera_checks": runtime["camera_checks"],
    },
    "task_success_verified": False,
    "hardware_run_verified": False,
}
args.report.parent.mkdir(parents=True, exist_ok=True)
with args.report.open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
