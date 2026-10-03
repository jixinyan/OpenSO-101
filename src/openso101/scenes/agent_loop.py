# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

"""The RGB-video to real2sim scene-agent loop.

The loop deliberately keeps model calls, Objaverse access, generated assets and
Isaac runtime checks behind small interfaces.  This lets the orchestration run
in a CPU-only checkout while the same state machine can be connected to
GPT-6 Astra, an Objaverse mirror, an asset-generation service and Isaac Lab.
"""

from __future__ import annotations

import json
import re
import tempfile
import objaverse
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bundle import export_bundle
from .catalog import AssetCatalog, search_categories
from .layout import diagnose_layout
from .model_client import ModelService
from .models import (
    Asset,
    Digest,
    Dimensions,
    Entity,
    Identifier,
    Physics,
    Pose,
    SceneSpec,
    Table,
    Task,
    Vector3,
    file_digest,
)
from .video import RGBVideoInput


class VideoObject(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    entity_id: Identifier
    label: str = Field(min_length=1)
    asset_query: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    dynamic: bool = True
    dimensions_m: Dimensions | None = None
    position_m: Vector3 | None = None


class VideoSceneDescription(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    instruction: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    objects: tuple[VideoObject, ...] = Field(min_length=1)
    table_visible: bool = True
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_objects(self):
        ids = [item.entity_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("视频识别出的 entity_id 必须唯一")
        return self


class AssetSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    entity_id: Identifier
    query: str = Field(min_length=1)
    limit: int = Field(default=8, ge=1, le=32)


class AssetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    uid: str = Field(pattern=r"^[a-f0-9]{32}$")
    name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    author: str = Field(min_length=1)
    local: bool = False
    category: str | None = None
    thumbnail_url: str | None = None

    @classmethod
    def from_asset(cls, asset: Asset) -> "AssetCandidate":
        return cls(
            uid=asset.uid,
            name=asset.name,
            source_url=asset.source_url,
            license=asset.license,
            author=asset.author,
            local=True,
        )


class AssetSearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    request: AssetSearchRequest
    candidates: tuple[AssetCandidate, ...] = ()
    provider: str = "objaverse"
    issues: tuple[str, ...] = ()


class GeneratedPart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    primitive: Literal["box", "cylinder", "sphere"]
    dimensions_m: Dimensions
    pose: Pose = Pose()
    color_rgba: tuple[int, int, int, int] = (180, 180, 180, 255)

    @model_validator(mode="after")
    def valid_color(self):
        if any(not 0 <= value <= 255 for value in self.color_rgba):
            raise ValueError("color_rgba 必须位于 0 到 255")
        return self


class DraftEntity(BaseModel):
    """An entity before an asset UID has necessarily been materialized."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    entity_id: Identifier
    asset_query: str = Field(min_length=1)
    asset_uid: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    primitive: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    generated_parts: tuple[GeneratedPart, ...] = Field(default=(), max_length=64)
    dimensions_m: Dimensions
    pose: Pose
    dynamic: bool = True
    physics: Physics = Physics(provenance="estimated")
    reset_translation_m: tuple[Vector3, Vector3] = ((0, 0, 0), (0, 0, 0))


class SceneDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    scene_id: Identifier
    table: Table = Table()
    robot_base: Pose = Pose()
    entities: tuple[DraftEntity, ...] = Field(min_length=1)
    task: Task
    reset_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unique_entities(self):
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("SceneDraft 中 entity_id 必须唯一")
        return self


class PlausibilityReview(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    approved: bool
    confidence: float = Field(ge=0, le=1)
    issues: tuple[str, ...] = ()
    suggested_changes: tuple[str, ...] = ()


class SO101ReadinessReview(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    approved: bool
    confidence: float = Field(ge=0, le=1)
    reachable: bool | None = None
    camera_visible: bool | None = None
    task_ready: bool = False
    issues: tuple[str, ...] = ()
    suggested_changes: tuple[str, ...] = ()


class SceneRevision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    revision: int = Field(ge=0)
    scene_sha256: Digest
    task_instruction: str = Field(min_length=1)
    instruction_source: Literal["user_context", "video_description"]


class AgentLoopResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    status: Literal["completed", "review_passed", "ready_for_collection", "needs_review", "checks_failed"]
    phase: str
    video: RGBVideoInput
    description: VideoSceneDescription
    asset_search: tuple[AssetSearchReport, ...]
    scene_sha256: str | None
    generated_asset_uids: tuple[str, ...] = ()
    bundle: str | None = None
    compiled_scene: str | None = None
    runtime_validation: dict[str, Any] | None = None
    static_diagnostics: dict[str, Any]
    physical: PlausibilityReview
    so101: SO101ReadinessReview
    iterations: tuple[dict[str, Any], ...] = ()
    scene_revisions: tuple[SceneRevision, ...] = ()
    pending_checks: tuple[str, ...] = ()


class ScenePlanner(Protocol):
    def describe(self, video: RGBVideoInput) -> VideoSceneDescription: ...

    def compose(
        self,
        video: RGBVideoInput,
        description: VideoSceneDescription,
        candidates: Sequence[AssetSearchReport],
    ) -> SceneDraft: ...

    def revise(
        self, video: RGBVideoInput, description: VideoSceneDescription,
        candidates: Sequence[AssetSearchReport], draft: SceneDraft, feedback: dict[str, Any],
    ) -> SceneDraft: ...

    def review_physical(
        self,
        video: RGBVideoInput,
        spec: SceneSpec,
        static_diagnostics: dict[str, Any],
    ) -> PlausibilityReview: ...

    def review_so101(
        self,
        video: RGBVideoInput,
        spec: SceneSpec,
        physical: PlausibilityReview,
    ) -> SO101ReadinessReview: ...


class AssetRetriever(Protocol):
    def search(self, request: AssetSearchRequest) -> Sequence[AssetCandidate]: ...

    def materialize(self, candidate: AssetCandidate) -> Asset: ...


class AssetGenerator(Protocol):
    def generate(self, entity: DraftEntity, catalog: AssetCatalog) -> Asset: ...


class ObjaverseRetriever:
    """Search local catalog first, then Objaverse LVIS annotations."""

    def __init__(self, catalog: AssetCatalog, *, online: bool = True):
        self.catalog = catalog
        self.online = online

    def search(self, request: AssetSearchRequest) -> Sequence[AssetCandidate]:
        tokens = {token for token in re.findall(r"\w+", request.query.casefold()) if len(token) > 1}
        local = []
        for asset in self.catalog.list():
            haystack = f"{asset.name} {asset.source_url}".casefold()
            score = sum(token in haystack for token in tokens)
            if score:
                local.append((score, AssetCandidate.from_asset(asset)))
        if local:
            return [item for _, item in sorted(local, key=lambda pair: (-pair[0], pair[1].uid))][:request.limit]
        if not self.online:
            return []

        candidates: list[AssetCandidate] = []
        categories = search_categories(request.query, request.limit)
        if not categories:
            categories = [category for token in sorted(tokens)
                          for category in search_categories(token, request.limit)]
        for category in categories:
            uids = category["uids"]
            annotations = objaverse.load_annotations(uids)
            for uid in uids:
                if not re.fullmatch(r"[a-f0-9]{32}", uid) or any(item.uid == uid for item in candidates):
                    continue
                metadata = annotations.get(uid, {})
                if not metadata.get("license"):
                    continue
                thumbnails = (metadata.get("thumbnails") or {}).get("images") or []
                thumbnail = next((image.get("url") for image in reversed(thumbnails)
                                  if str(image.get("url", "")).startswith(("http://", "https://"))), None)
                candidates.append(
                    AssetCandidate(
                        uid=uid,
                        name=str(metadata.get("name") or category["category"]),
                        source_url=str(metadata.get("viewerUrl") or f"https://objaverse.org/objects/{uid}"),
                        license=str(metadata.get("license") or "unknown"),
                        author=str((metadata.get("user") or {}).get("displayName") or "unknown"),
                        category=category["category"],
                        thumbnail_url=thumbnail,
                    )
                )
                if len(candidates) >= request.limit:
                    return candidates
        return candidates

    def materialize(self, candidate: AssetCandidate) -> Asset:
        return self.catalog.fetch(candidate.uid)


class TrimeshAssetGenerator:
    """Build declarative meshes without executing model-generated code."""

    def generate(self, entity: DraftEntity, catalog: AssetCatalog) -> Asset:
        import trimesh

        import numpy as np
        from scipy.spatial.transform import Rotation

        parts = entity.generated_parts or (GeneratedPart(
            primitive=(entity.primitive or "box").casefold(), dimensions_m=entity.dimensions_m,
        ),)
        meshes = []
        for part in parts:
            x, y, z = part.dimensions_m
            if part.primitive == "box":
                mesh = trimesh.creation.box(extents=(x, y, z))
            elif part.primitive == "cylinder":
                mesh = trimesh.creation.cylinder(radius=1, height=1, sections=32)
                mesh.apply_scale((x / 2, y / 2, z))
            else:
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=1)
                mesh.apply_scale((x / 2, y / 2, z / 2))
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(part.pose.quaternion_wxyz, scalar_first=True).as_matrix()
            transform[:3, 3] = part.pose.position
            mesh.apply_transform(transform)
            mesh.visual.face_colors = part.color_rgba
            meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        # Recipes are Z-up; glTF is Y-up. The compiler converts back to Z-up.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
        recipe = [part.model_dump() for part in parts]
        with tempfile.TemporaryDirectory(prefix="openso101-generated-", dir=catalog.root) as temp:
            path = Path(temp) / "model.glb"
            mesh.export(path)
            return catalog.import_glb(
                path,
                uid=file_digest(path)[:32],
                metadata={
                    "name": "generated_mesh",
                    "viewerUrl": "generated://trimesh/recipe",
                    "license": "MIT",
                    "user": {"displayName": "OpenSO-101"},
                    "recipe": recipe,
                    "source_up_axis": "Y",
                },
            )


class AstraScenePlanner:
    """GPT-6 Astra planner using structured JSON at every loop boundary."""

    def __init__(self, service: ModelService, *, image_limit: int = 4):
        self.service = service
        if not isinstance(image_limit, int) or isinstance(image_limit, bool) or not 1 <= image_limit <= 32:
            raise ValueError("image_limit 必须位于 1 到 32")
        self.image_limit = image_limit

    def describe(self, video: RGBVideoInput) -> VideoSceneDescription:
        description = self.service.complete(
            system=(
                "你是 GPT-6 Astra 的 real2sim 感知模块。根据 RGB 视频帧识别桌面、"
                "可操作物体和 SO-101 任务意图。只返回 schema JSON；尺寸和位置不确定时"
                "使用保守估计，并把不确定性写入 notes。"
            ),
            prompt=json.dumps(video.model_dump(), ensure_ascii=False),
            schema=VideoSceneDescription,
            images=video.model_images(limit=self.image_limit),
        )
        if video.context.instruction and description.instruction != video.context.instruction:
            # 用户任务文本保持原文，模型感知字段保留其生成结果。
            description = description.model_copy(update={"instruction": video.context.instruction})
        return description

    def revise(
        self, video: RGBVideoInput, description: VideoSceneDescription,
        candidates: Sequence[AssetSearchReport], draft: SceneDraft, feedback: dict[str, Any],
    ) -> SceneDraft:
        return self.service.complete(
            system=(
                "你是 GPT-6 Astra 的 real2sim repair planner。根据上一版场景和失败检查报告"
                "只修改必要字段，修复碰撞、支撑、任务目标、尺寸或 SO-101 可采集性问题。"
                "资产只能使用候选 UID，或使用声明式 primitive/generated_parts；只返回完整 schema JSON。"
            ),
            prompt=json.dumps({
                "video": video.model_dump(), "description": description.model_dump(),
                "asset_search": [item.model_dump() for item in candidates],
                "draft": draft.model_dump(), "feedback": feedback,
            }, ensure_ascii=False),
            schema=SceneDraft,
            images=video.model_images(limit=self.image_limit),
        )

    def compose(
        self,
        video: RGBVideoInput,
        description: VideoSceneDescription,
        candidates: Sequence[AssetSearchReport],
    ) -> SceneDraft:
        return self.service.complete(
            system=(
                "你是 GPT-6 Astra 的 real2sim 场景编排模块。根据视频识别结果和 Objaverse "
                "候选资产生成 SO-101 桌面场景。只能选择候选中的 asset_uid；找不到合适资产时"
                "将 asset_uid 设为 null，并设置 primitive 或 generated_parts（box/cylinder/sphere、"
                "尺寸、局部姿态和 RGBA）以便生成可追踪的近似资产。"
                "所有长度用米，姿态 quaternion 为 wxyz，物体必须在桌面上方。"
            ),
            prompt=json.dumps({
                "video": video.model_dump(),
                "description": description.model_dump(),
                "asset_search": [item.model_dump() for item in candidates],
            }, ensure_ascii=False),
            schema=SceneDraft,
            images=video.model_images(limit=self.image_limit),
        )

    def review_physical(
        self,
        video: RGBVideoInput,
        spec: SceneSpec,
        static_diagnostics: dict[str, Any],
    ) -> PlausibilityReview:
        return self.service.complete(
            system=(
                "你是 GPT-6 Astra 的 physical plausibility reviewer。检查场景几何、物体"
                "支撑关系、质量摩擦、碰撞体和 reset 范围。只返回 schema JSON；不能把尚未"
                "运行 Isaac 的结果说成已验证。"
            ),
            prompt=json.dumps({"scene": spec.model_dump(), "static": static_diagnostics}, ensure_ascii=False),
            schema=PlausibilityReview,
            images=video.model_images(limit=self.image_limit),
        )

    def review_so101(
        self,
        video: RGBVideoInput,
        spec: SceneSpec,
        physical: PlausibilityReview,
    ) -> SO101ReadinessReview:
        return self.service.complete(
            system=(
                "你是 GPT-6 Astra 的 SO-101 embodied-data reviewer。检查机器人工作空间、"
                "夹爪可达性、相机可见性、任务目标和是否能安全采集演示。没有 Isaac 或真机"
                "测量时把 reachable/camera_visible 设为 null，并在 issues 中说明。"
            ),
            prompt=json.dumps({"scene": spec.model_dump(), "physical": physical.model_dump()}, ensure_ascii=False),
            schema=SO101ReadinessReview,
            images=video.model_images(),
        )


class AgentLoopError(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(f"agent loop [{phase}] {message}")
        self.phase = phase


class Real2SimAgentLoop:
    """Run perception -> retrieval -> composition -> review -> SO-101 checks."""

    def __init__(
        self,
        catalog: AssetCatalog,
        planner: ScenePlanner,
        retriever: AssetRetriever | None = None,
        generator: AssetGenerator | None = None,
        physics_runtime_check: Callable[[SceneSpec], dict[str, Any]] | None = None,
        so101_runtime_check: Callable[[SceneSpec], dict[str, Any]] | None = None,
        max_revisions: int = 2,
    ):
        self.catalog = catalog
        self.planner = planner
        self.retriever = retriever or ObjaverseRetriever(catalog)
        self.generator = generator
        self.physics_runtime_check = physics_runtime_check
        self.so101_runtime_check = so101_runtime_check
        if not isinstance(max_revisions, int) or isinstance(max_revisions, bool) or not 0 <= max_revisions <= 5:
            raise ValueError("max_revisions 必须位于 0 到 5")
        self.max_revisions = max_revisions

    def run(self, video: RGBVideoInput, *, output: Path | None = None) -> AgentLoopResult:
        try:
            description = self.planner.describe(video)
            instruction = video.context.instruction or description.instruction
            if description.instruction != instruction:
                description = description.model_copy(update={"instruction": instruction})
            requests = [AssetSearchRequest(entity_id=obj.entity_id, query=obj.asset_query)
                        for obj in description.objects]
            search_reports = tuple(
                self._search_report(request) for request in requests
            )
            draft = self.planner.compose(video, description, search_reports)
            iterations: list[dict[str, Any]] = []
            scene_revisions: list[SceneRevision] = []
            physical = None
            so101 = None
            spec = None
            static: dict[str, Any] = {}
            generated_asset_uids: tuple[str, ...] = ()
            for iteration in range(self.max_revisions + 1):
                if draft.task.instruction != instruction:
                    draft = draft.model_copy(update={
                        "task": draft.task.model_copy(update={"instruction": instruction}),
                    })
                spec, generated_asset_uids = self._materialize(draft, search_reports)
                scene_revisions.append(SceneRevision(
                    revision=iteration, scene_sha256=spec.digest(),
                    task_instruction=spec.task.instruction,
                    instruction_source="user_context" if video.context.instruction else "video_description",
                ))
                static = diagnose_layout(spec, self.catalog)
                if static["status"] != "static_checks_passed":
                    feedback = {"phase": "static", "diagnostics": static}
                    iterations.append(feedback)
                    if iteration < self.max_revisions and hasattr(self.planner, "revise"):
                        draft = self.planner.revise(video, description, search_reports, draft, feedback)
                        continue
                    raise AgentLoopError("physical", json.dumps(static, ensure_ascii=False))
                physics_runtime = self.physics_runtime_check(spec) if self.physics_runtime_check else None
                if physics_runtime is not None:
                    static = {**static, "runtime": physics_runtime}
                physical = self.planner.review_physical(video, spec, static)
                if physics_runtime is not None:
                    physical = physical.model_copy(update={
                        "approved": bool(physical.approved and physics_runtime.get("approved", True)),
                        "issues": physical.issues + tuple(str(item) for item in physics_runtime.get("issues", ())),
                        "suggested_changes": physical.suggested_changes + tuple(
                            str(item) for item in physics_runtime.get("suggested_changes", ())
                        ),
                    })
                if not physical.approved:
                    feedback = {"phase": "physical", "review": physical.model_dump(), "static": static}
                    iterations.append(feedback)
                    if iteration < self.max_revisions and hasattr(self.planner, "revise"):
                        draft = self.planner.revise(video, description, search_reports, draft, feedback)
                        continue
                so101 = self.planner.review_so101(video, spec, physical)
                if self.so101_runtime_check:
                    so101_runtime = self.so101_runtime_check(spec)
                    so101 = so101.model_copy(update={
                        "approved": bool(so101.approved and so101_runtime.get("approved", True)),
                        "reachable": so101_runtime.get("reachable", so101.reachable),
                        "camera_visible": so101_runtime.get("camera_visible", so101.camera_visible),
                        "task_ready": bool(so101.task_ready and so101_runtime.get("task_ready", True)),
                        "issues": so101.issues + tuple(str(item) for item in so101_runtime.get("issues", ())),
                        "suggested_changes": so101.suggested_changes + tuple(
                            str(item) for item in so101_runtime.get("suggested_changes", ())
                        ),
                    })
                if so101.approved and so101.task_ready and physical.approved:
                    break
                feedback = {"phase": "so101", "review": so101.model_dump(), "physical": physical.model_dump()}
                iterations.append(feedback)
                if iteration < self.max_revisions and hasattr(self.planner, "revise"):
                    draft = self.planner.revise(video, description, search_reports, draft, feedback)
                    continue
                break
            assert spec is not None and physical is not None and so101 is not None
            bundle = None
            if output is not None:
                bundle = str(export_bundle(spec, self.catalog, output))
            status = "completed" if physical.approved and so101.approved and so101.task_ready else "needs_review"
            return AgentLoopResult(
                status=status,
                phase="complete" if status == "completed" else "review",
                video=video,
                description=description,
                asset_search=search_reports,
                scene_sha256=spec.digest(),
                generated_asset_uids=generated_asset_uids,
                bundle=bundle,
                static_diagnostics=static,
                physical=physical,
                so101=so101,
                iterations=tuple(iterations),
                scene_revisions=tuple(scene_revisions),
                pending_checks=tuple(static.get("pending_checks", ())),
            )
        except AgentLoopError:
            raise
        except Exception as exc:
            raise AgentLoopError("orchestration", str(exc)) from exc

    def _materialize(
        self,
        draft: SceneDraft,
        reports: Sequence[AssetSearchReport],
    ) -> tuple[SceneSpec, tuple[str, ...]]:
        candidates = {candidate.uid: candidate for report in reports for candidate in report.candidates}
        entities: list[Entity] = []
        generated_asset_uids: list[str] = []
        for draft_entity in draft.entities:
            asset = None
            if draft_entity.asset_uid:
                candidate = candidates.get(draft_entity.asset_uid)
                if candidate is None:
                    raise AgentLoopError("retrieval", f"模型选择了未检索到的资产：{draft_entity.asset_uid}")
                asset = self.retriever.materialize(candidate)
            elif self.generator:
                asset = self.generator.generate(draft_entity, self.catalog)
                generated_asset_uids.append(asset.uid)
            else:
                raise AgentLoopError("asset_generation", f"缺少资产且没有 generator：{draft_entity.asset_query}")
            entities.append(Entity(
                entity_id=draft_entity.entity_id,
                asset_uid=asset.uid,
                asset_sha256=asset.sha256,
                dimensions_m=draft_entity.dimensions_m,
                pose=draft_entity.pose,
                dynamic=draft_entity.dynamic,
                physics=draft_entity.physics,
                reset_translation_m=draft_entity.reset_translation_m,
            ))
        return SceneSpec(
            scene_id=draft.scene_id,
            table=draft.table,
            robot_base=draft.robot_base,
            entities=tuple(entities),
            task=draft.task,
            reset_seed=draft.reset_seed,
        ), tuple(generated_asset_uids)

    def _search_report(self, request: AssetSearchRequest) -> AssetSearchReport:
        candidates = tuple(self.retriever.search(request))
        return AssetSearchReport(
            request=request,
            candidates=candidates,
            issues=("no Objaverse candidate; planner must provide generated_parts",) if not candidates else (),
        )
