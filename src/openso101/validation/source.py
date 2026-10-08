import ast
import configparser
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile

from openso101.scenes.models import file_digest


def _module_name(path, source):
    parts = path.relative_to(source).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _namespace(tree):
    names = set()

    def visit(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Import):
                names.update(item.asname or item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.update(item.asname or item.name for item in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(item.id for target in targets for item in ast.walk(target)
                             if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store))
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
    visit(tree.body)
    return names


def check_source(repo: Path, output: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("源码检查需要禁止 CUDA")
    repo = repo.resolve()
    output = output.resolve()
    if not output.is_relative_to(repo / "outputs"):
        raise ValueError("源码检查结果需要保存到项目 outputs 目录")
    output.mkdir(parents=True, exist_ok=False)
    source = repo / "src"
    files = sorted(path for folder in ("src", "scripts", "tests")
                   for path in (repo / folder).rglob("*.py") if "__pycache__" not in path.parts)
    trees = {path: ast.parse(path.read_text(), filename=str(path)) for path in files}
    modules = {_module_name(path, source): path for path in files if path.is_relative_to(source)}
    namespaces = {name: _namespace(trees[path]) for name, path in modules.items()}
    imports = 0
    for path, tree in trees.items():
        compile(tree, str(path), "exec")
        package = ".".join(path.relative_to(source).parts[:-1]) if path.is_relative_to(source) else ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                requested = [item.name for item in node.names if item.name.startswith("openso101")]
            elif isinstance(node, ast.ImportFrom):
                relative = "." * node.level + (node.module or "")
                name = importlib.util.resolve_name(relative, package) if node.level and package else relative
                requested = [name] if name.startswith("openso101") else []
                if requested and name in namespaces and "__getattr__" not in namespaces[name]:
                    for item in node.names:
                        if item.name != "*" and item.name not in namespaces[name] and f"{name}.{item.name}" not in modules:
                            raise ImportError(f"项目 import 的名称不存在：{path}:{node.lineno} {name}.{item.name}")
            else:
                continue
            for name in requested:
                if name not in modules:
                    raise ModuleNotFoundError(f"项目 import 路径不存在：{path}:{node.lineno} {name}")
                imports += 1
    for name, path in modules.items():
        for parent in path.parents:
            if parent == source:
                break
            if not (parent / "__init__.py").is_file():
                raise FileNotFoundError(f"Python package 缺少 __init__.py：{name} {parent}")
    shell_files = sorted((repo / "scripts").rglob("*.sh"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
    environment = {**os.environ, "PYTHONPATH": str(source), "OPENSO101_SKIP_ISAAC": "1"}
    cli_groups = ("envs", "rl", "il", "sim2sim", "sim2real", "scenes", "validate")
    for group in cli_groups:
        completed = subprocess.run([sys.executable, "-m", "openso101.cli.main", group, "--help"],
                                   cwd=repo, env=environment, check=True, capture_output=True, text=True)
        (output / f"{group}.help.txt").write_text(completed.stdout)
    report = {"status": "source_and_cli_verified", "python_files": len(files), "modules": len(modules),
              "project_imports": imports, "shell_files": len(shell_files), "cli_groups": list(cli_groups),
              "source_files": {path.relative_to(repo).as_posix(): file_digest(path) for path in files + shell_files},
              "gpu_tests_started": False, "native_physics_verified": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def check_wheel(repo: Path, wheel: Path, output: Path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("wheel 检查需要禁止 CUDA")
    repo = repo.resolve()
    output = output.resolve()
    if not output.is_relative_to(repo / "outputs"):
        raise ValueError("wheel 检查结果需要保存到项目 outputs 目录")
    output.mkdir(parents=True, exist_ok=False)
    source = repo / "src"
    expected = {path.relative_to(source).as_posix(): file_digest(path)
                for path in (source / "openso101").rglob("*.py") if "__pycache__" not in path.parts}
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    with ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError("wheel 的 ZIP CRC 检查未通过")
        actual = {name: hashlib.sha256(archive.read(name)).hexdigest()
                  for name in archive.namelist() if name.startswith("openso101/") and name.endswith(".py")}
        if actual != expected:
            raise ValueError("wheel 中的 Python 模块与当前源码内容不一致")
        metadata_name, = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = message_from_bytes(archive.read(metadata_name))
        if metadata["Name"] != project["name"] or metadata["Version"] != project["version"]:
            raise ValueError("wheel 的名称或版本与 pyproject.toml 不一致")
        entry_name, = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(entry_name).decode())
        if dict(entry_points["console_scripts"]) != project["scripts"]:
            raise ValueError("wheel 的命令入口与 pyproject.toml 不一致")
    report = {"status": "wheel_source_and_entry_points_verified", "wheel_sha256": file_digest(wheel),
              "modules": len(actual), "name": project["name"], "version": project["version"],
              "source_files": expected, "gpu_tests_started": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
