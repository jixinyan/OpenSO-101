import argparse
import json
import os
from pathlib import Path

from pydantic import Field

from openso101.scenes.agent.model_client import ModelService, load_codex_runtime_config
from openso101.scenes.models import Model


class InstructionRecord(Model):
    instruction: str
    object_names: tuple[str, ...] = Field(min_length=1)
    require_released: bool


parser = argparse.ArgumentParser()
parser.add_argument("--model-config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
config = load_codex_runtime_config(args.model_config)
key_environment = "OPENSO101_MODEL_CHECK_KEY"
if config.bearer_token:
    os.environ[key_environment] = config.bearer_token
service = ModelService(base_url=config.base_url, model=config.model, wire_api=config.wire_api,
                       api_key_env=key_environment, reasoning_effort="low", max_requests=1, max_retries=0)
instruction = "将红色方块放入托盘，打开夹爪，等待物体稳定。"
record = service.complete(system="提取用户任务，instruction 保留原文。",
                          prompt=instruction, schema=InstructionRecord)
if record.instruction != instruction or not record.require_released:
    raise RuntimeError("模型提取结果没有保留任务条件")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"result": record.model_dump(), "requests": service.requests},
                                 ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"model": service.model, "status": service.requests[0]["status"],
                  "usage": service.requests[0]["usage"], "output": str(args.output)}))
