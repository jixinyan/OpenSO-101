# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import base64
import hashlib
import json
import math
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import httpx
from httpx_sse import EventSource
from pydantic import BaseModel


@dataclass(frozen=True)
class ModelService:
    base_url: str
    model: str
    api_key_env: str = "SCENE_MODEL_API_KEY"
    timeout_seconds: float = 120
    wire_api: str = "chat"
    max_retries: int = 2
    reasoning_effort: str = "xhigh"
    reasoning_summary: str = "auto"
    max_requests: int | None = None
    requests: list[dict] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self):
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or not isinstance(self.base_url, str)
            or not self.base_url.startswith(("https://", "http://"))
        ):
            raise ValueError("模型名称和 HTTP 服务地址不能为空")
        if self.wire_api not in ("chat", "responses"):
            raise ValueError("wire_api 必须为 chat 或 responses")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于零")
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool) or not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries 必须位于 0 到 5")
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh"):
            raise ValueError("reasoning_effort 必须为 low、medium、high 或 xhigh")
        if self.max_requests is not None and (not isinstance(self.max_requests, int)
                                             or isinstance(self.max_requests, bool) or self.max_requests <= 0):
            raise ValueError("max_requests 必须为正整数")

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
        images: Sequence[str | Path] = (),
    ) -> BaseModel:
        system = system + "\n仅返回符合以下 schema 的 JSON：\n" + schema_instruction(schema)
        headers = {}
        # 本地兼容服务可以接受匿名请求；配置密钥后发送 Authorization。
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env, "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        endpoint = self.base_url.rstrip("/")
        if self.wire_api == "chat" and not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        elif self.wire_api == "responses" and not endpoint.endswith("/responses"):
            endpoint += "/responses"
        if self.wire_api == "responses":
            response_content: list[dict] = [{"type": "input_text", "text": prompt}]
            response_content.extend(
                {"type": "input_image", "image_url": _image_url(image)} for image in images
            )
            request_body = {
                "model": self.model,
                # 网关使用 developer input，并要求通过 SSE 完成 Responses 请求。
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": system}]},
                    {"role": "user", "content": response_content},
                ],
                "reasoning": {"effort": self.reasoning_effort, "summary": self.reasoning_summary},
                "text": {"verbosity": "low"},
                "stream": True,
                "store": False,
            }
        else:
            if images:
                content: list[dict] = [{"type": "text", "text": prompt}]
                for image in images:
                    content.append({"type": "image_url", "image_url": {"url": _image_url(image)}})
                user_message: dict = {"role": "user", "content": content}
            else:
                # 文本请求保留兼容服务接受的字段格式。
                user_message = {"role": "user", "content": prompt}
            request_body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    user_message,
                ],
                "response_format": {"type": "json_object"},
            }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            completion = None
            for attempt in range(self.max_retries + 1):
                if self.max_requests is not None and len(self.requests) >= self.max_requests:
                    raise RuntimeError("模型请求数量达到预算上限")
                record = {"attempt": attempt, "model": self.model, "schema": schema.__name__,
                          "wire_api": self.wire_api, "status": "requested", "usage": None,
                          "request_sha256": hashlib.sha256(json.dumps(request_body, sort_keys=True).encode()).hexdigest()}
                self.requests.append(record)
                started = time.monotonic()
                try:
                    if self.wire_api == "responses":
                        with client.stream("POST", endpoint, headers=headers, json=request_body) as response:
                            response.raise_for_status()
                            completion = _parse_response_stream(response)
                    else:
                        response = client.post(endpoint, headers=headers, json=request_body)
                        response.raise_for_status()
                        completion = response.json()
                    record.update(status="response_received", elapsed_seconds=time.monotonic() - started,
                                  usage=completion.get("usage"),
                                  response_sha256=hashlib.sha256(json.dumps(completion, sort_keys=True).encode()).hexdigest())
                    result = (_parse_response_output(completion, schema) if self.wire_api == "responses"
                              else _parse_chat_output(completion, schema))
                    record["status"] = "validated"
                    break
                except httpx.HTTPStatusError as exc:
                    record.update(status="http_error", http_status=exc.response.status_code,
                                  elapsed_seconds=time.monotonic() - started)
                    retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
                    if not retryable or attempt >= self.max_retries:
                        raise
                except httpx.TransportError:
                    record.update(status="transport_error", elapsed_seconds=time.monotonic() - started)
                    if attempt >= self.max_retries:
                        raise
                except (RuntimeError, ValueError, TypeError):
                    record.update(status="invalid_response", elapsed_seconds=time.monotonic() - started)
                    raise
                time.sleep(min(2 ** attempt, 8))
            assert completion is not None
        return result


def _parse_chat_output(completion: object, schema: type[BaseModel]) -> BaseModel:
    try:
        choice = completion["choices"][0]
        finish_reason = choice.get("finish_reason", "stop")
        message = choice["message"]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError("模型服务返回缺少 choices[0].message") from exc
    if not isinstance(message, dict):
        raise ValueError("模型服务返回的 message 需要 JSON 对象")
    if finish_reason not in (None, "stop"):
        raise ValueError(f"模型输出未完成：{finish_reason}")
    if message.get("refusal"):
        raise ValueError("模型服务拒绝了本次请求")
    content = message.get("content")
    if content is None:
        raise ValueError("模型服务返回的 message.content 为空")
    return schema.model_validate_json(content) if isinstance(content, str) else schema.model_validate(content)


@dataclass(frozen=True)
class CodexRuntimeConfig:
    """Non-secret model settings plus an in-process bearer token."""

    model: str
    base_url: str
    wire_api: str = "chat"
    bearer_token: str | None = field(default=None, repr=False)
    reasoning_effort: str = "xhigh"
    reasoning_summary: str = "auto"


def load_codex_runtime_config(path: Path) -> CodexRuntimeConfig:
    """Read the local Codex TOML format without printing credentials."""
    import tomllib

    if not path.is_file():
        raise FileNotFoundError(path)
    data = tomllib.loads(path.read_text())
    provider_name = data.get("model_provider")
    providers = data.get("model_providers") or {}
    provider = providers.get(provider_name, {})
    if not isinstance(provider, dict):
        raise ValueError("Codex 配置的 model provider 无效")
    token = provider.get("experimental_bearer_token")
    if token is not None and not isinstance(token, str):
        raise ValueError("Codex bearer token 必须是字符串")
    return CodexRuntimeConfig(
        model=str(data.get("model") or "gpt-6-astra"),
        base_url=str(provider.get("base_url") or ""),
        wire_api=str(provider.get("wire_api") or "chat"),
        bearer_token=token,
        reasoning_effort=str(data.get("model_reasoning_effort") or "xhigh"),
        reasoning_summary=str(data.get("model_reasoning_summary") or "auto"),
    )


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


def _parse_response_output(completion: object, schema: type[BaseModel]) -> BaseModel:
    if not isinstance(completion, dict):
        raise ValueError("Responses API 返回不是对象")
    if completion.get("status") not in (None, "completed"):
        raise ValueError(f"模型输出未完成：{completion.get('status')}")
    content = completion.get("output_text")
    if not isinstance(content, str):
        output = completion.get("output", [])
        if isinstance(output, list):
            text_parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for part in item.get("content", ()):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            content = "".join(text_parts)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Responses API 返回缺少 output_text")
    return schema.model_validate_json(content)


def _parse_response_stream(response: object) -> dict:
    headers = getattr(response, "headers", {})
    content_type = str(headers.get("content-type", ""))
    if "text/event-stream" not in content_type:
        response.read()
        return response.json()
    deltas: list[str] = []
    completed: dict | None = None
    for message in EventSource(response).iter_sse():
        if not message.data or message.data == "[DONE]":
            continue
        event = message.json()
        if not isinstance(event, dict):
            raise ValueError("Responses 事件需要 JSON 对象")
        event_type = event.get("type")
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            deltas.append(event["delta"])
        elif event_type == "response.completed":
            response_value = event.get("response")
            if isinstance(response_value, dict):
                completed = response_value
                break
        elif event_type in ("response.failed", "response.incomplete", "error"):
            raise RuntimeError(f"Responses 流终止：{event_type}")
    if completed is None:
        raise RuntimeError("Responses 流缺少 response.completed")
    if deltas:
        completed["output_text"] = "".join(deltas)
    return completed


def schema_instruction(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)
