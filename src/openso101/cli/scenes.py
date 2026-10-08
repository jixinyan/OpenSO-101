# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(args):
    from openso101.scenes.bundle import export_bundle, validate_layout, verify_bundle
    from openso101.scenes.assets.catalog import AssetCatalog, search_categories
    from openso101.scenes.layout import diagnose_layout
    from openso101.scenes.models import Entity, Pose, SceneSpec, Task

    if args.command == "search":
        result = search_categories(args.query, args.limit)
    elif args.command == "verify":
        result = {"scene_id": verify_bundle(args.bundle).scene_id, "status": "bundle_verified"}
    elif args.command == "diagnose":
        catalog = AssetCatalog(args.catalog)
        from openso101.scenes.models import SceneSpec

        result = diagnose_layout(SceneSpec.read(args.scene_file), catalog, clearance_m=args.clearance)
    else:
        catalog = AssetCatalog(args.catalog)
        if args.command == "fetch":
            result = catalog.fetch(args.uid).model_dump()
        elif args.command == "list":
            result = [asset.model_dump() for asset in catalog.list()]
        elif args.command == "create":
            asset = catalog.read(args.uid)
            height = args.dimensions[2]
            spec = SceneSpec(
                scene_id=args.scene_id,
                entities=(Entity(
                    entity_id="object", asset_uid=asset.uid, asset_sha256=asset.sha256,
                    dimensions_m=tuple(args.dimensions), pose=Pose(position=(0.25, 0, height / 2)),
                ),),
                task=Task(task_id="place_object", object_id="object", goal_position_m=(0.25, 0.15, height / 2)),
            )
            validate_layout(spec, catalog)
            with args.output.open("x") as stream:
                stream.write(spec.model_dump_json(indent=2))
            result = {"scene_file": str(args.output.resolve()), "scene_sha256": spec.digest()}
        elif args.command == "validate":
            result = validate_layout(SceneSpec.read(args.scene_file), catalog)
        elif args.command == "bundle":
            path = export_bundle(SceneSpec.read(args.scene_file), catalog, args.output)
            result = {"bundle": str(path), "status": "layout_valid"}
        else:
            raise ValueError(f"未知命令：{args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def add_subparsers(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="为实际资产目录生成 embedding 索引")
    index.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    index.add_argument("--model-revision")
    index.set_defaults(func=_index_assets)
    retrieval = sub.add_parser("retrieve", help="使用 embedding 检索实际资产")
    retrieval.add_argument("query")
    retrieval.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    retrieval.add_argument("--index", type=Path, required=True)
    retrieval.add_argument("--limit", type=int, default=8)
    retrieval.add_argument("--minimum-similarity", type=float, default=.25)
    retrieval.set_defaults(func=_retrieve_assets)
    runtime = sub.add_parser("validate-runtime", help="Check resets, parallel simulation and optional cameras")
    runtime.add_argument("scene", type=Path)
    runtime.add_argument("--output", type=Path, required=True)
    runtime.add_argument("--num-envs", type=int, default=4)
    runtime.add_argument("--steps", type=int, default=200)
    runtime.add_argument("--resets", type=int, default=100)
    runtime.add_argument("--cameras", action="store_true")
    runtime.set_defaults(func=_validate_runtime)
    prepare = sub.add_parser("prepare", help="编译场景并执行 Isaac 双相机与并行环境检查")
    prepare.add_argument("bundle", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--num-envs", type=int, default=4)
    prepare.add_argument("--steps", type=int, default=200)
    prepare.add_argument("--resets", type=int, default=100)
    prepare.set_defaults(func=_prepare)
    save = sub.add_parser("save", help="Save a scene revision with an expected base revision")
    save.add_argument("scene_file", type=Path)
    save.add_argument("--store", type=Path, default=Path("outputs/scenes.sqlite"))
    save.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    save.add_argument("--expected-revision", type=int, required=True)
    save.add_argument("--reason", required=True)
    save.set_defaults(func=_save)
    read = sub.add_parser("read", help="Export a saved scene revision")
    read.add_argument("scene_id")
    read.add_argument("--store", type=Path, default=Path("outputs/scenes.sqlite"))
    read.add_argument("--revision", type=int)
    read.add_argument("--output", type=Path, required=True)
    read.set_defaults(func=_read)
    template = sub.add_parser("robotwin-task", help="Create an SO-101 task bundle from RoboTwin assets")
    template.add_argument("name", choices=("mouse_pad", "stapler_pad", "pillbottle_pad", "object_scale", "stack_blocks"))
    template.add_argument("--source", type=Path, required=True)
    template.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--license", required=True)
    template.add_argument("--author", required=True)
    template.set_defaults(func=_robotwin_task)
    generate = sub.add_parser("generate", help="Generate a scene proposal through an OpenAI-compatible service")
    generate.add_argument("--instruction", required=True)
    generate.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    generate.add_argument("--asset-index", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--base-url", default=os.environ.get("SCENE_MODEL_BASE_URL"))
    generate.add_argument("--model", default=os.environ.get("SCENE_MODEL_NAME"))
    generate.add_argument("--api-key-env", default="SCENE_MODEL_API_KEY")
    generate.add_argument("--codex-config", type=Path)
    generate.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"))
    generate.add_argument("--max-model-requests", type=int, default=12)
    generate.add_argument("--max-revisions", type=int, default=3)
    generate.add_argument("--runtime-output", type=Path)
    generate.add_argument("--runtime-budget-seconds", type=float, default=600)
    generate.add_argument("--capability-probes", type=Path)
    generate.set_defaults(func=_generate)
    edit = sub.add_parser("edit", help="根据自然语言修改已有场景并保存独立版本")
    edit.add_argument("bundle", type=Path)
    edit.add_argument("--instruction", required=True)
    edit.add_argument("--output", type=Path, required=True)
    edit.add_argument("--edit-file", type=Path)
    edit.add_argument("--catalog", type=Path)
    edit.add_argument("--store", type=Path)
    edit.add_argument("--expected-revision", type=int)
    edit.add_argument("--base-url", default=os.environ.get("SCENE_MODEL_BASE_URL"))
    edit.add_argument("--model", default=os.environ.get("SCENE_MODEL_NAME"))
    edit.add_argument("--api-key-env", default="SCENE_MODEL_API_KEY")
    edit.add_argument("--codex-config", type=Path)
    edit.add_argument("--runtime-output", type=Path)
    edit.set_defaults(func=_edit)
    bddl = sub.add_parser("read-bddl", help="使用官方 BDDL 库读取完整外部任务")
    bddl.add_argument("source", type=Path)
    bddl.add_argument("--output", type=Path, required=True)
    bddl.add_argument("--scene-file", type=Path)
    bddl.add_argument("--binding", type=Path)
    bddl.add_argument("--bundle-output", type=Path)
    bddl.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    bddl.set_defaults(func=_read_bddl)
    calibrate = sub.add_parser("calibrate-video", help="使用实际测量点与独立验证点标定视频相机")
    calibrate.add_argument("video", type=Path)
    calibrate.add_argument("--measurements", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.set_defaults(func=_calibrate_video)
    metric = sub.add_parser("metric-layout", help="根据标定相机与已知参考高度恢复实际物体位置")
    metric.add_argument("scene_file", type=Path)
    metric.add_argument("--calibration", type=Path, required=True)
    metric.add_argument("--observations", type=Path, required=True)
    metric.add_argument("--video", type=Path, required=True)
    metric.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    metric.add_argument("--output", type=Path, required=True)
    metric.set_defaults(func=_metric_layout)
    preview = sub.add_parser("preview", help="生成包含实际 mesh 与官方机器人模型的交互式场景预览")
    preview.add_argument("bundle", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--robot-model", type=Path)
    preview.add_argument("--joint-positions", nargs=6, type=float)
    preview.set_defaults(func=_preview)
    import_asset = sub.add_parser("import", help="Import a user-provided GLB into the asset catalog")
    import_asset.add_argument("file", type=Path)
    import_asset.add_argument("--name", required=True)
    import_asset.add_argument("--source", required=True)
    import_asset.add_argument("--license", required=True)
    import_asset.add_argument("--author", required=True)
    import_asset.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    import_asset.set_defaults(func=_import_asset)
    for command in ("import-usd", "import-robotwin"):
        importer = sub.add_parser(command)
        importer.add_argument("source", type=Path)
        importer.add_argument("--license", required=True)
        importer.add_argument("--author", required=True)
        importer.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
        if command == "import-usd":
            importer.add_argument("--prim", required=True)
            importer.add_argument("--name", required=True)
        else:
            importer.add_argument("--model-id", type=int, default=0)
        importer.set_defaults(func=_import_external)
    inspect = sub.add_parser("inspect", help="Measure asset geometry and list capability evidence")
    inspect.add_argument("uid")
    inspect.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    inspect.add_argument("--probe", type=Path)
    inspect.add_argument("--output", type=Path)
    inspect.set_defaults(func=_inspect)
    layout = sub.add_parser("layout", help="Solve tabletop footprint placement and reset clearance")
    layout.add_argument("scene_file", type=Path)
    layout.add_argument("--output", type=Path, required=True)
    layout.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    layout.add_argument("--locked", nargs="*", default=[])
    layout.add_argument("--clearance", type=float, default=0.01)
    layout.set_defaults(func=_layout)
    diagnose = sub.add_parser(
        "diagnose",
        help="Run CPU-only collision and relation checks for a scene",
    )
    diagnose.add_argument("scene_file", type=Path)
    diagnose.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    diagnose.add_argument("--clearance", type=float, default=0.0)
    diagnose.set_defaults(func=_run)
    agent = sub.add_parser(
        "agent-loop",
        help="Run RGB-video -> GPT-6 Astra -> Objaverse/generated assets -> SO-101 review",
    )
    agent.add_argument("--video", required=True, help="RGB video source path or URI")
    agent.add_argument("--instruction", default="", help="Optional task instruction to preserve through reconstruction")
    agent.add_argument("--frame", dest="frame_paths", action="append", default=[],
                       help="Sampled RGB frame path; repeat for multiple frames (optional)")
    agent.add_argument("--fps", type=float, help="Required when --frame is supplied")
    agent.add_argument("--frame-count", type=int, help="Required when --frame is supplied")
    agent.add_argument("--width", type=int, help="Required when --frame is supplied")
    agent.add_argument("--height", type=int, help="Required when --frame is supplied")
    agent.add_argument("--sample-count", type=int, default=8, help="Frames to decode when --frame is omitted")
    agent.add_argument("--max-revisions", type=int, default=2, help="Maximum Astra repair iterations after review failures")
    agent.add_argument("--max-model-requests", type=int, default=12)
    agent.add_argument("--timeout-seconds", type=float, default=300, help="Per-request Astra timeout")
    agent.add_argument("--image-limit", type=int, default=4, help="Maximum RGB frames attached to each Astra request")
    agent.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"),
                       help="Override the Codex reasoning effort")
    agent.add_argument("--frames-dir", type=Path, help="Persistent directory for automatically sampled frames")
    agent.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
    agent.add_argument("--offline-objaverse", action="store_true",
                       help="Skip online Objaverse metadata/downloads and let Astra generate missing assets")
    agent.add_argument("--output", type=Path, required=True, help="Output portable scene bundle")
    agent.add_argument("--runtime-output", type=Path, help="保存自动编译与 Isaac 双相机运行检查")
    agent.add_argument("--runtime-num-envs", type=int, default=4)
    agent.add_argument("--runtime-steps", type=int, default=200)
    agent.add_argument("--runtime-resets", type=int, default=100)
    agent.add_argument("--base-url", default=os.environ.get("SCENE_MODEL_BASE_URL"))
    agent.add_argument("--model", default=os.environ.get("SCENE_MODEL_NAME"))
    agent.add_argument("--api-key-env", default="SCENE_MODEL_API_KEY")
    agent.add_argument(
        "--codex-config",
        type=Path,
        default=Path(os.environ["CODEX_CONFIG"]) if os.environ.get("CODEX_CONFIG") else None,
        help="Optional Codex TOML config; defaults to the local LitchiAgent runtime config when present",
    )
    agent.set_defaults(func=_agent_loop)
    compile_parser = sub.add_parser("compile", help="Convert bundle to USD using Isaac Sim")
    compile_parser.add_argument("bundle", type=Path)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.set_defaults(func=_compile)
    search = sub.add_parser("search", help="Search Objaverse LVIS categories")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.set_defaults(func=_run)
    verify = sub.add_parser("verify", help="Verify portable scene bundle")
    verify.add_argument("bundle", type=Path)
    verify.set_defaults(func=_run)
    for command in ("fetch", "list", "create", "validate", "bundle"):
        child = sub.add_parser(command)
        child.add_argument("--catalog", type=Path, default=Path("outputs/assets"))
        child.set_defaults(func=_run)
        if command in ("fetch", "create"):
            child.add_argument("uid")
        if command == "create":
            child.add_argument("--scene-id", required=True)
            child.add_argument("--dimensions", type=float, nargs=3, required=True, metavar=("X_M", "Y_M", "Z_M"))
        if command in ("create", "bundle"):
            child.add_argument("--output", type=Path, required=True)
        if command in ("validate", "bundle"):
            child.add_argument("scene_file", type=Path)


def _compile(args):
    from openso101.scenes.bundle import verify_bundle
    from openso101.scenes.isaaclab.usd import verify_compilation

    verify_bundle(args.bundle)
    if args.output.exists():
        raise FileExistsError(args.output)

    subprocess.run([
        sys.executable, "-m", "openso101.scenes.isaaclab.worker",
        str(args.bundle.resolve()), str(args.output.resolve()),
    ], check=True)
    report = verify_compilation(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _generate(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.agent.model_client import ModelService, load_codex_runtime_config
    from openso101.scenes.agent.workflow import JobLimits, generate_scene_job

    config = load_codex_runtime_config(args.codex_config) if args.codex_config is not None else None
    if config is not None and config.bearer_token and not os.environ.get(args.api_key_env, "").strip():
        os.environ[args.api_key_env] = config.bearer_token
    base_url = args.base_url or (config.base_url if config else None)
    model = args.model or (config.model if config else None)
    if not base_url or not model:
        raise ValueError("请设置 --base-url 和 --model，或对应的 SCENE_MODEL 环境变量")
    if args.output.exists():
        raise FileExistsError(args.output)
    service = ModelService(base_url, model, args.api_key_env, wire_api=config.wire_api if config else "chat",
                           reasoning_effort=args.reasoning_effort or (config.reasoning_effort if config else "xhigh"))
    limits = JobLimits(max_model_requests=args.max_model_requests, max_revisions=args.max_revisions,
                       max_runtime_seconds=args.runtime_budget_seconds)
    result = generate_scene_job(args.instruction, AssetCatalog(args.catalog), service, args.output,
                                limits=limits, runtime_output=args.runtime_output, asset_index=args.asset_index,
                                capability_probes=json.loads(args.capability_probes.read_text())
                                if args.capability_probes else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _index_assets(args):
    from openso101.scenes.assets.index import AssetIndex
    from openso101.scenes.assets.catalog import AssetCatalog

    report = AssetIndex.build(AssetCatalog(args.catalog), args.output, model=args.embedding_model,
                              revision=args.model_revision)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _retrieve_assets(args):
    from openso101.scenes.assets.index import AssetIndex
    from openso101.scenes.assets.catalog import AssetCatalog

    report = AssetIndex(AssetCatalog(args.catalog), args.index).search(
        args.query, limit=args.limit, minimum_similarity=args.minimum_similarity)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _agent_loop(args):
    from openso101.scenes.agent.loop import (
        AstraScenePlanner,
        RGBVideoInput,
        Real2SimAgentLoop,
        ObjaverseRetriever,
        TrimeshAssetGenerator,
    )
    from openso101.scenes.video import sample_rgb_video
    from openso101.scenes.video import SceneContext
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.agent.model_client import ModelService, load_codex_runtime_config

    if args.output.exists():
        raise FileExistsError(args.output)

    if args.runtime_output is not None:
        if args.runtime_output.exists():
            raise FileExistsError(args.runtime_output)
        if args.runtime_output.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("场景准备输出需要位于源 bundle 目录以外")
        if min(args.runtime_num_envs, args.runtime_steps, args.runtime_resets) <= 0:
            raise ValueError("环境数量、步骤数量和 reset 次数必须大于零")

    # LitchiAgent keeps the active Astra provider in this runtime config.  We
    # discover it for the local checkout while still allowing an explicit path
    # or the normal SCENE_MODEL_* environment variables to take precedence.
    config = None
    config_path = args.codex_config
    if config_path is None:
        candidate = Path("/mnt/data/users/jixin/workspace/code/LitchiAgent/runtime/codex/config.toml")
        if candidate.is_file():
            config_path = candidate
    if config_path is not None:
        config = load_codex_runtime_config(config_path)
        if config.bearer_token and not os.environ.get(args.api_key_env, "").strip():
            os.environ[args.api_key_env] = config.bearer_token
    base_url = args.base_url or (config.base_url if config else None)
    model = args.model or (config.model if config else "gpt-6-astra")
    wire_api = config.wire_api if config else "chat"
    reasoning_effort = args.reasoning_effort or (config.reasoning_effort if config else "xhigh")
    reasoning_summary = config.reasoning_summary if config else "auto"
    if not base_url:
        raise ValueError("请设置 --base-url、SCENE_MODEL_BASE_URL 或提供 Codex runtime config")
    if args.frame_paths:
        if any(value is None for value in (args.fps, args.frame_count, args.width, args.height)):
            raise ValueError("使用 --frame 时必须同时提供 --fps、--frame-count、--width 和 --height")
        video = RGBVideoInput(
            source=args.video, frame_count=args.frame_count, fps=args.fps,
            width=args.width, height=args.height, frame_paths=tuple(args.frame_paths),
            context=SceneContext(instruction=args.instruction),
        )
    else:
        frames_dir = args.frames_dir or args.output.parent / f".{args.output.name}.frames"
        video = sample_rgb_video(
            Path(args.video), frames_dir, count=args.sample_count,
            context=SceneContext(instruction=args.instruction),
        )
    catalog = AssetCatalog(args.catalog)
    loop = Real2SimAgentLoop(
        catalog,
        AstraScenePlanner(ModelService(
            base_url, model, args.api_key_env,
            timeout_seconds=args.timeout_seconds, wire_api=wire_api,
            reasoning_effort=reasoning_effort, reasoning_summary=reasoning_summary,
            max_requests=args.max_model_requests,
        ), image_limit=args.image_limit),
        retriever=ObjaverseRetriever(catalog, online=not args.offline_objaverse),
        generator=TrimeshAssetGenerator(),
        max_revisions=args.max_revisions,
    )
    result = loop.run(video, output=args.output)
    if args.runtime_output is not None:
        from openso101.scenes.isaaclab.preparation import prepare_scene

        prepared = prepare_scene(
            Path(result.bundle), args.runtime_output, num_envs=args.runtime_num_envs,
            steps=args.runtime_steps, resets=args.runtime_resets,
        )
        result = result.model_copy(update={
            "status": "simulation_ready",
            "phase": "simulation",
            "compiled_scene": prepared["compiled_scene"], "runtime_validation": prepared["runtime"],
            "pending_checks": tuple(prepared["pending_checks"]),
        })
        with (args.runtime_output / "agent_loop_result.json").open("x") as stream:
            stream.write(result.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))


def _import_asset(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.models import file_digest

    if args.file.suffix.lower() != ".glb":
        raise ValueError("资产导入需要 GLB 文件")
    asset = AssetCatalog(args.catalog).import_glb(args.file, uid=file_digest(args.file)[:32], metadata={
        "name": args.name, "viewerUrl": args.source, "license": args.license,
        "user": {"displayName": args.author},
    })
    print(asset.model_dump_json(indent=2))


def _inspect(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.assets.inspection import inspect_asset

    from openso101.scenes.assets.capabilities import GeometryProbe

    probe = GeometryProbe.model_validate_json(args.probe.read_text()) if args.probe else None
    report = inspect_asset(AssetCatalog(args.catalog), args.uid, probe=probe)
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _edit(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.editor.editing import SceneEdit, interpret_scene_edit, revise_bundle
    from openso101.scenes.agent.model_client import ModelService, load_codex_runtime_config
    from openso101.scenes.editor.store import SceneStore

    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.store is None) != (args.expected_revision is None):
        raise ValueError("版本保存需要同时提供 --store 与 --expected-revision")
    requests = []
    catalog = AssetCatalog(args.catalog) if args.catalog else None
    if args.edit_file is not None:
        edit = SceneEdit.model_validate_json(args.edit_file.read_text())
        if edit.instruction != args.instruction:
            raise ValueError("修改文件与用户描述不一致")
    else:
        config = load_codex_runtime_config(args.codex_config) if args.codex_config else None
        if config and config.bearer_token and not os.environ.get(args.api_key_env, "").strip():
            os.environ[args.api_key_env] = config.bearer_token
        base_url = args.base_url or (config.base_url if config else None)
        model = args.model or (config.model if config else None)
        if not base_url or not model:
            raise ValueError("自然语言修改需要配置实际模型服务")
        service = ModelService(base_url, model, args.api_key_env, wire_api=config.wire_api if config else "chat",
                               reasoning_effort=config.reasoning_effort if config else "xhigh", max_requests=3)
        edit = interpret_scene_edit(args.bundle, args.instruction, service, catalog=catalog)
        requests = service.requests
    result = revise_bundle(args.bundle, edit, args.output,
                           store=SceneStore(args.store) if args.store else None,
                           expected_revision=args.expected_revision, model_requests=requests, catalog=catalog)
    if args.runtime_output is not None:
        from openso101.scenes.isaaclab.preparation import prepare_scene

        result["runtime"] = prepare_scene(args.output, args.runtime_output)
        result["pending_checks"] = result["runtime"]["pending_checks"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _read_bddl(args):
    from openso101.scenes.bddl import BDDLBinding, bind_bddl_problem, read_bddl_problem
    from openso101.scenes.models import SceneSpec

    if args.bundle_output and not args.binding:
        raise ValueError("BDDL bundle 导出需要完整场景与 binding")
    if (args.scene_file is None) != (args.binding is None):
        raise ValueError("BDDL 目标绑定需要同时提供场景与 binding")
    if args.binding:
        result = bind_bddl_problem(args.source, BDDLBinding.model_validate_json(args.binding.read_text()),
                                   SceneSpec.read(args.scene_file), args.output)
    else:
        result = read_bddl_problem(args.source)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.bundle_output:
        from openso101.scenes.bundle import export_bundle
        from openso101.scenes.assets.catalog import AssetCatalog

        export_bundle(SceneSpec.read(args.scene_file), AssetCatalog(args.catalog), args.bundle_output,
                      bddl_report=result, provenance={"operation": "bind_bddl", "task_success_verified": False})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _calibrate_video(args):
    from openso101.scenes.metric_video import CameraMeasurements, calibrate_camera

    report = calibrate_camera(CameraMeasurements.model_validate_json(args.measurements.read_text()), args.video)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _metric_layout(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.layout import diagnose_layout
    from openso101.scenes.metric_video import ObjectImageMeasurement, apply_metric_positions, verify_calibration
    from openso101.scenes.models import SceneSpec, file_digest

    if args.output.exists():
        raise FileExistsError(args.output)
    observations = tuple(ObjectImageMeasurement.model_validate(item) for item in json.loads(args.observations.read_text()))
    calibration = json.loads(args.calibration.read_text())
    verify_calibration(calibration, args.video)
    scene = apply_metric_positions(SceneSpec.read(args.scene_file), observations, calibration)
    diagnostics = diagnose_layout(scene, AssetCatalog(args.catalog))
    if diagnostics["status"] != "static_checks_passed":
        raise ValueError(f"视频恢复布局未通过检查：{diagnostics}")
    with args.output.open("x") as stream:
        stream.write(scene.model_dump_json(indent=2))
    report = {"scene_sha256": scene.digest(), "status": "metric_layout_created",
              "source_scene_sha256": SceneSpec.read(args.scene_file).digest(),
              "video_sha256": file_digest(args.video), "calibration_sha256": file_digest(args.calibration),
              "observations_sha256": file_digest(args.observations),
              "observations": [item.model_dump(mode="json") for item in observations],
              "scene_reconstruction_verified": False, "task_success_verified": False,
              "pending_checks": diagnostics["pending_checks"]}
    with args.output.with_suffix(".provenance.json").open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))


def _preview(args):
    from openso101.scenes.editor.preview import export_preview

    if args.joint_positions is not None and args.robot_model is None:
        raise ValueError("机器人关节位置需要同时指定官方 MJCF 文件")
    report = export_preview(args.bundle, args.output, robot_model=args.robot_model, joint_positions=args.joint_positions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _layout(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.layout import solve_layout
    from openso101.scenes.models import SceneSpec

    solved = solve_layout(SceneSpec.read(args.scene_file), AssetCatalog(args.catalog),
                          locked=args.locked, clearance_m=args.clearance)
    with args.output.open("x") as stream:
        stream.write(solved.model_dump_json(indent=2))
    print(json.dumps({"scene_file": str(args.output.resolve()), "scene_sha256": solved.digest()}))


def _import_external(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.assets.importers import import_robotwin, import_usd

    catalog = AssetCatalog(args.catalog)
    if args.command == "import-usd":
        asset = import_usd(catalog, args.source, args.prim, name=args.name, license=args.license, author=args.author)
    else:
        asset = import_robotwin(catalog, args.source, args.model_id, license=args.license, author=args.author)
    print(asset.model_dump_json(indent=2))


def _robotwin_task(args):
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.templates import create_robotwin_task

    path = create_robotwin_task(args.name, args.source, AssetCatalog(args.catalog), args.output,
                               license=args.license, author=args.author)
    print(json.dumps({"bundle": str(path), "task_id": args.name}))


def _save(args):
    from openso101.scenes.bundle import validate_layout
    from openso101.scenes.assets.catalog import AssetCatalog
    from openso101.scenes.models import SceneSpec
    from openso101.scenes.editor.store import SceneStore

    spec = SceneSpec.read(args.scene_file)
    validate_layout(spec, AssetCatalog(args.catalog))
    revision = SceneStore(args.store).save(spec, expected_revision=args.expected_revision, reason=args.reason)
    print(json.dumps({"scene_id": spec.scene_id, "revision": revision, "scene_sha256": spec.digest()}))


def _read(args):
    from openso101.scenes.editor.store import SceneStore

    revision, spec = SceneStore(args.store).read(args.scene_id, args.revision)
    with args.output.open("x") as stream:
        stream.write(spec.model_dump_json(indent=2))
    print(json.dumps({"scene_id": spec.scene_id, "revision": revision, "scene_file": str(args.output.resolve())}))


def _validate_runtime(args):
    from openso101.scenes.isaaclab.usd import verify_compilation

    compilation = verify_compilation(args.scene)
    if args.output.exists():
        raise FileExistsError(args.output)
    command = [sys.executable, "-u", "-m", "openso101.scenes.isaaclab.validation_worker",
               str(args.scene.resolve()), str(args.output.resolve()), "--num-envs", str(args.num_envs),
               "--steps", str(args.steps), "--resets", str(args.resets)]
    if args.cameras:
        command.append("--cameras")
    subprocess.run(command, check=True)
    if not args.output.is_file():
        raise RuntimeError(
            "Isaac runtime worker 未生成报告；请检查 Isaac Sim 启动日志（常见原因是首次运行需要接受 NVIDIA EULA）"
        )
    report = json.loads(args.output.read_text())
    if report["scene_sha256"] != compilation["scene_sha256"] or report["status"] != "runtime_verified":
        raise ValueError("运行报告与请求场景不匹配")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _prepare(args):
    from openso101.scenes.isaaclab.preparation import prepare_scene

    report = prepare_scene(args.bundle, args.output, num_envs=args.num_envs, steps=args.steps, resets=args.resets)
    print(json.dumps(report, ensure_ascii=False, indent=2))
