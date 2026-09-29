# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import base64
import json
import math
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx
from pydantic import BaseModel


@dataclass(frozen=True)
class ModelService:
    base_url: str
    model: str
    api_key_env: str = "SCENE_MODEL_API_KEY"
    timeout_seconds: float = 120

    def __post_init__(self):
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or not isinstance(self.base_url, str)
            or not self.base_url.startswith(("https://", "http://"))
        ):
            raise ValueError("模型名称和 HTTP 服务地址不能为空")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于零")

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
        images: Sequence[str | Path] = (),
    ) -> BaseModel:
        system = system + "\nReturn only JSON conforming to this schema:\n" + schema_instruction(schema)
        headers = {}
        # Empty/missing credentials are valid for local OpenAI-compatible
        # servers.  Only send an Authorization header when a non-empty key is
        # actually configured; this also avoids an unhelpful KeyError before
        # the HTTP request is made.
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env, "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": _image_url(image)}})
            user_message: dict = {"role": "user", "content": content}
        else:
            # Keep the text-only shape for existing OpenAI-compatible servers.
            user_message = {"role": "user", "content": prompt}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                endpoint,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        user_message,
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            completion = response.json()
        try:
            choice = completion["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            message = choice["message"]
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError("模型服务返回缺少 choices[0].message") from exc
        if not isinstance(message, dict):
            raise ValueError("模型服务返回的 message 不是对象")
        if finish_reason not in (None, "stop"):
            raise ValueError(f"模型输出未完成：{finish_reason}")
        if message.get("refusal"):
            raise ValueError("模型服务拒绝了本次请求")
        content = message.get("content")
        if content is None:
            raise ValueError("模型服务返回的 message.content 为空")
        if isinstance(content, str):
            return schema.model_validate_json(content)
        # A few compatible servers decode JSON mode into an object before
        # serialising the response.  Accept that representation as well.
        return schema.model_validate(content)


def _image_url(image: str | Path) -> str:
    """Return an OpenAI-compatible image URL for a local frame or URL."""
    value = str(image)
    if value.startswith(("data:", "http://", "https://")):
        return value
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def schema_instruction(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)
