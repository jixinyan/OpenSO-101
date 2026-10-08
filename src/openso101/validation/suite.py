import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from jsonpointer import resolve_pointer
from pydantic import Field, model_validator

from openso101.scenes.models import Digest, Identifier, Model, file_digest


class ReportRequirement(Model):
    report: str = Field(min_length=1)
    pointer: str
    equals: bool | float | str | None = None
    minimum: float | None = None

    @model_validator(mode="after")
    def explicit_requirement(self):
        if (self.equals is None) == (self.minimum is None):
            raise ValueError("验收要求必须指定 equals 或 minimum")
        if self.pointer and not self.pointer.startswith("/"):
            raise ValueError("报告字段需要使用 JSON Pointer")
        return self


class ValidationStage(Model):
    name: Identifier
    resource: Literal["cpu", "gpu"]
    command: tuple[str, ...] = Field(min_length=1)
    depends_on: tuple[Identifier, ...] = ()
    requirements: tuple[ReportRequirement, ...] = Field(min_length=1)
    inputs: dict[str, Digest] = Field(default_factory=dict)


class ValidationSuite(Model):
    schema_version: Literal[1] = 1
    physical_gpu: Literal[2] = 2
    source_files: tuple[str, ...] = Field(min_length=1)
    stages: tuple[ValidationStage, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def ordered_dependencies(self):
        seen = {}
        for stage in self.stages:
            if stage.name in seen or set(stage.depends_on) - set(seen):
                raise ValueError("验收步骤需要唯一名称，并且依赖必须先声明")
            if stage.resource == "cpu" and any(seen[name] == "gpu" for name in stage.depends_on):
                raise ValueError("CPU 前置检查不能依赖 GPU 结果")
            if "-c" in stage.command or any("\n" in item for item in stage.command):
                raise ValueError("验收命令需要使用保存的脚本或模块入口")
            if stage.resource == "gpu" and stage.command[:3] != ("bash", "scripts/run_native_python.sh", "2"):
                raise ValueError("GPU 步骤必须使用项目的单张 GPU 2 启动入口")
            seen[stage.name] = stage.resource
        if not {"cpu", "gpu"}.issubset(seen.values()):
            raise ValueError("完整验收需要 CPU 与 GPU 两种步骤")
        return self


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"验收路径超出指定目录：{relative}")
    return path


def _sources(repo: Path, suite: ValidationSuite) -> dict:
    files = {}
    for relative in suite.source_files:
        path = _inside(repo, relative)
        if path.is_dir():
            selected = [item for item in path.rglob("*") if item.is_file()
                        and "__pycache__" not in item.parts and item.suffix in (".py", ".sh", ".json", ".toml")]
        elif path.is_file():
            selected = [path]
        else:
            raise FileNotFoundError(path)
        for item in selected:
            verified = _inside(repo, item.relative_to(repo).as_posix())
            files[item.relative_to(repo).as_posix()] = file_digest(verified)
    return files


def run_suite(manifest: Path, output: Path, *, phase: Literal["cpu", "gpu"], repo: Path):
    if phase not in ("cpu", "gpu"):
        raise ValueError("统一验收需要指定 cpu 或 gpu 阶段")
    repo = repo.resolve()
    output = output.resolve()
    if not output.is_relative_to(repo / "outputs"):
        raise ValueError("统一验收结果需要位于仓库 outputs 目录")
    suite = ValidationSuite.model_validate_json(manifest.read_text())
    snapshot = _sources(repo, suite)
    receipt_path = output / "suite.json"
    if phase == "cpu":
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        receipt = {"schema_version": 1, "manifest_sha256": file_digest(manifest), "source_files": snapshot,
                   "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                   "physical_gpu": suite.physical_gpu, "created_at": datetime.now(UTC).isoformat(),
                   "status": "cpu_running", "stages": [], "gpu_tests_started": False,
                   "full_v2_verified": False, "hardware_run_verified": False}
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt["manifest_sha256"] != file_digest(manifest) or receipt["source_files"] != snapshot:
            raise ValueError("统一 GPU 验收需要使用通过 CPU 检查的相同源码与配置")
        if receipt["gpu_tests_started"]:
            raise ValueError("GPU 验收已经执行；保留本次记录并创建明确的新批次")
        records = {item["name"]: item for item in receipt["stages"]}
        for stage in suite.stages:
            if stage.resource == "cpu":
                if stage.name not in records or records[stage.name]["status"] != "verified":
                    raise ValueError(f"统一 GPU 验收的 CPU 前置检查未通过：{stage.name}")
                _verify_stage_inputs(stage, repo)
                if records[stage.name]["reports"] != _accept_reports(stage, output):
                    raise ValueError("CPU 验收报告的内容或 SHA256 已改变")
        receipt.update(status="gpu_running", gpu_tests_started=True)

    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    save()
    for stage in suite.stages:
        if stage.resource != phase:
            continue
        records = {item["name"]: item for item in receipt["stages"]}
        if any(name not in records or records[name]["status"] != "verified" for name in stage.depends_on):
            receipt["status"] = "dependency_not_verified"
            save()
            raise ValueError(f"验收步骤依赖尚未通过：{stage.name}")
        _verify_stage_inputs(stage, repo)
        command = [item.replace("{output}", str(output)) for item in stage.command]
        if command[0] == "{python}":
            import sys

            command[0] = sys.executable
        environment = os.environ.copy()
        if phase == "cpu":
            environment.update(CUDA_VISIBLE_DEVICES="", OPENSO101_SKIP_ISAAC="1")
        log = output / f"{stage.name}.log"
        record = {"name": stage.name, "resource": phase, "command": command, "status": "running",
                  "started_at": datetime.now(UTC).isoformat(), "inputs": stage.inputs}
        receipt["stages"].append(record)
        save()
        with log.open("x") as stream:
            completed = subprocess.run(command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record.update(exit_code=completed.returncode, log_sha256=file_digest(log),
                      completed_at=datetime.now(UTC).isoformat())
        if completed.returncode:
            record["status"] = "process_failed"
            receipt["status"] = "failed"
            save()
            raise RuntimeError(f"验收步骤失败：{stage.name}；日志：{log}")
        record["status"] = "requirements_not_verified"
        receipt["status"] = "requirements_not_verified"
        save()
        record["reports"] = _accept_reports(stage, output)
        record["status"] = "verified"
        receipt["status"] = f"{phase}_running"
        save()
    receipt["status"] = "cpu_verified" if phase == "cpu" else "declared_suite_verified"
    receipt["completed_at"] = datetime.now(UTC).isoformat()
    save()
    return receipt


def _verify_stage_inputs(stage, repo):
    for relative, expected in stage.inputs.items():
        if file_digest(_inside(repo, relative)) != expected:
            raise ValueError(f"验收输入文件 SHA256 已改变：{relative}")


def _accept_reports(stage, output):
    reports = {}
    for requirement in stage.requirements:
        path = _inside(output, requirement.report)
        report = json.loads(path.read_text())
        actual = resolve_pointer(report, requirement.pointer)
        if requirement.minimum is not None:
            if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                    or not math.isfinite(actual) or actual < requirement.minimum):
                raise ValueError(f"{stage.name} 未达到 {requirement.pointer} >= {requirement.minimum}：{actual}")
        elif (actual != requirement.equals or (isinstance(requirement.equals, bool) and type(actual) is not bool)
              or (isinstance(requirement.equals, float) and isinstance(actual, bool))):
            raise ValueError(f"{stage.name} 的 {requirement.pointer} 与验收要求不一致：{actual}")
        reports[requirement.report] = file_digest(path)
    return reports
