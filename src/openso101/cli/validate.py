import argparse
import json
from pathlib import Path


def _cmd_source(args):
    from openso101.validation.source import check_source

    report = check_source(Path.cwd(), args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "source_files"}, ensure_ascii=False))
    return 0


def _cmd_run(args):
    from openso101.validation.suite import run_suite

    report = run_suite(args.manifest, args.output, phase=args.phase, repo=Path.cwd())
    print(json.dumps({"status": report["status"], "stages": len(report["stages"]),
                      "gpu_tests_started": report["gpu_tests_started"]}, ensure_ascii=False))
    return 0


def _cmd_package(args):
    from openso101.validation.source import check_wheel

    report = check_wheel(Path.cwd(), args.wheel, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "source_files"}, ensure_ascii=False))
    return 0


def add_subparsers(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source", help="检查源码、项目 import、Shell 与全部命令入口")
    source.add_argument("--output", type=Path, required=True)
    source.set_defaults(func=_cmd_source)
    package = sub.add_parser("package", help="检查实际 wheel 的源码、版本和命令入口")
    package.add_argument("--wheel", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.set_defaults(func=_cmd_package)
    run = sub.add_parser("run", help="执行保存来源与报告的 CPU 或 GPU 验证")
    run.add_argument("manifest", type=Path)
    run.add_argument("--phase", choices=("cpu", "gpu"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=_cmd_run)
