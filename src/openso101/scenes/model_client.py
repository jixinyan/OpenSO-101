# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import json
import os
from dataclasses import dataclass

import httpx
from pydantic import BaseModel


@dataclass(frozen=True)
class ModelService:
    base_url: str
    model: str
    api_key_env: str = "SCENE_MODEL_API_KEY"
    timeout_seconds: float = 120

    def __post_init__(self):
        if not self.model.strip() or not self.base_url.startswith(("https://", "http://")):
            raise ValueError("模型名称和 HTTP 服务地址不能为空")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于零")

    def complete(self, *, system: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
        headers = {}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            completion = response.json()
        choice = completion["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError(f"模型输出未完成：{choice['finish_reason']}")
        message = choice["message"]
        if message.get("refusal"):
            raise ValueError("模型服务拒绝了本次请求")
        return schema.model_validate_json(message["content"])


def schema_instruction(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)
