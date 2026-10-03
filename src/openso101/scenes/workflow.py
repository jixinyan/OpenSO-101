import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from .bundle import export_bundle
from .catalog import AssetCatalog
from .layout import diagnose_layout
from .models import Model, SceneSpec, file_digest
from .program import compile_program, extract_intent


class JobLimits(Model):
    max_model_requests: int = Field(default=12, ge=1)
    max_revisions: int = Field(default=3, ge=0, le=3)
    max_candidates: int = Field(default=64, ge=1)
    max_entities: int = Field(default=16, ge=1)
    max_runtime_seconds: float = Field(default=600, gt=0)


def generate_scene_job(instruction: str, catalog: AssetCatalog, service, output: Path,
                       *, limits: JobLimits | None = None, runtime_output: Path | None = None,
                       asset_index: Path | None = None):
    limits = limits or JobLimits()
    if service.requests:
        raise ValueError("场景任务需要独立 ModelService 请求记录")
    service = replace(service, max_requests=limits.max_model_requests)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    job = {"schema_version": 1, "instruction": instruction, "status": "started",
           "created_at": datetime.now(UTC).isoformat(), "limits": limits.model_dump(),
           "tool_calls": [], "model_requests": service.requests, "downloaded_bytes": 0,
           "collection_ready": False, "dataset_verified": False,
           "workflow_source_sha256": file_digest(Path(__file__))}

    def save(status):
        job.update(status=status, elapsed_seconds=time.monotonic() - started)
        (output / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n")

    def tool(name, inputs, operation):
        record = {"tool": name, "input": inputs, "status": "running"}
        job["tool_calls"].append(record)
        tool_started = time.monotonic()
        save(job["status"])
        try:
            result = operation()
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, message=str(error),
                          elapsed_seconds=time.monotonic() - tool_started)
            save("failed")
            raise
        value = result.model_dump(mode="json") if isinstance(result, Model) else result
        record.update(status="completed", result=value, elapsed_seconds=time.monotonic() - tool_started,
                      result_sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        save(job["status"])
        return result

    save("started")
    intent = tool("extract_intent", {"instruction": instruction}, lambda: extract_intent(instruction, service))
    (output / "task_intent.json").write_text(intent.model_dump_json(indent=2))
    job["intent_sha256"] = intent.digest()
    if intent.clarifications:
        job["clarifications"] = list(intent.clarifications)
        save("needs_input")
        return job
    save("intent_ready")
    assets = catalog.list()
    candidate_uids = {}
    if len(intent.objects) > limits.max_entities:
        save("failed")
        raise ValueError("任务实体数量达到预算上限")
    if asset_index is not None:
        from .asset_index import AssetIndex

        index = None

        def load_index():
            nonlocal index
            index = AssetIndex(catalog, asset_index)
            return index.manifest

        tool("load_asset_index", {"index": str(asset_index.resolve())}, load_index)
        job["asset_index_sha256"] = file_digest(asset_index / "index.json")
        for item in intent.objects:
            candidates = tool("search_assets", {"entity_id": item.entity_id, "query": item.asset_query,
                                                  "limit": min(8, limits.max_candidates)},
                              lambda: index.search(item.asset_query, limit=min(8, limits.max_candidates)))
            if not candidates:
                job["clarifications"] = [f"当前资产目录缺少匹配资产：{item.label}"]
                save("needs_input")
                return job
            candidate_uids[item.entity_id] = {entry["uid"] for entry in candidates}
        selected = set().union(*candidate_uids.values())
        assets = [item for item in assets if item.uid in selected]
    if not assets or len(assets) > limits.max_candidates:
        save("failed")
        raise ValueError("资产候选数量必须位于当前预算范围")
    asset_records = [item.model_dump() for item in assets]
    tool("read_catalog", {"catalog": str(catalog.root)}, lambda: asset_records)
    save("assets_ready")
    payload = {"intent": intent.model_dump(), "assets": asset_records,
               "entity_candidates": {name: sorted(uids) for name, uids in candidate_uids.items()}}
    spec = None
    program = None
    diagnostics = None
    try:
        for revision in range(limits.max_revisions + 1):
            spec = tool("compose_scene", {"revision": revision, "intent_sha256": intent.digest()},
                        lambda: service.complete(
                            system=("生成 SO-101 桌面 SceneSpec。仅使用提供的资产 UID 与 SHA256。"
                                    "task.instruction 必须逐字复制 intent.instruction，禁止添加、改写或附加说明。"
                                    "操作顺序和条件由 TaskProgram 保存，无需追加到 task.instruction。"
                                    "完整保留全部 TaskIntent 最终目标和稳定释放要求。"
                                    "保留实体 identifier、尺寸和已确认的默认值，不能删减任务条件。"
                                    "质量摩擦估计使用 estimated provenance；所有长度使用米，Z 轴向上。"
                                    "根据实际 diagnostics 修复场景，不能更改 TaskIntent。"),
                            prompt=json.dumps(payload, ensure_ascii=False), schema=SceneSpec))
            if len(spec.entities) > limits.max_entities:
                raise ValueError("场景实体数量达到预算上限")
            if candidate_uids and any(item.entity_id not in candidate_uids
                                      or item.asset_uid not in candidate_uids[item.entity_id] for item in spec.entities):
                raise ValueError("场景实体必须使用其检索结果中的资产")
            program = compile_program(intent, spec)
            diagnostics = tool("validate_layout", {"scene_sha256": spec.digest()},
                               lambda: diagnose_layout(spec, catalog))
            if diagnostics["status"] == "static_checks_passed":
                break
            payload.update(scene=spec.model_dump(), diagnostics=diagnostics)
        if diagnostics["status"] != "static_checks_passed":
            raise RuntimeError("场景静态检查在预算内未通过")
        save("layout_valid")
        provenance = {"model": service.model, "model_requests": service.requests,
                      "intent_sha256": intent.digest(), "layout": diagnostics,
                      "job_limits": limits.model_dump(), "downloaded_bytes": 0,
                      "asset_index_sha256": job.get("asset_index_sha256"),
                      "entity_candidates": {name: sorted(uids) for name, uids in candidate_uids.items()}}
        bundle = export_bundle(spec, catalog, output / "bundle", intent=intent, program=program,
                               provenance=provenance)
        job.update(scene_sha256=spec.digest(), task_program_sha256=program.digest(), bundle=str(bundle),
                   pending_checks=diagnostics["pending_checks"])
        if runtime_output is not None:
            from .preparation import prepare_scene

            prepared = tool("prepare_scene", {"bundle": str(bundle), "num_envs": 4, "steps": 200, "resets": 100},
                            lambda: prepare_scene(bundle, runtime_output, timeout_seconds=limits.max_runtime_seconds))
            job.update(compiled_scene=prepared["compiled_scene"], runtime_report_sha256=prepared["runtime_report_sha256"],
                       pending_checks=prepared["pending_checks"])
            save("simulation_ready")
        else:
            save("layout_valid")
        return job
    except Exception:
        save("failed")
        raise
