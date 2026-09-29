# Copyright (c) 2026, Jixin Yan
# SPDX-License-Identifier: MIT

import base64
import json
import math
import mimetypes
import os
import time
from dataclasses import dataclass, field
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
    wire_api: str = "chat"
    max_retries: int = 2
    reasoning_effort: str = "xhigh"
    reasoning_summary: str = "auto"

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
                # Codex's Responses wire sends the system instruction as a
                # developer input item.  The gateway requires streaming even
                # for structured one-shot calls, so collect its SSE deltas
                # below before validating the final JSON.
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
                # Keep the text-only shape for existing OpenAI-compatible servers.
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
                try:
                    if self.wire_api == "responses":
                        with client.stream("POST", endpoint, headers=headers, json=request_body) as response:
                            response.raise_for_status()
                            completion = _parse_response_stream(response)
                    else:
                        response = client.post(endpoint, headers=headers, json=request_body)
                        response.raise_for_status()
                        completion = response.json()
                    break
                except httpx.HTTPStatusError as exc:
                    retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
                    if not retryable or attempt >= self.max_retries:
                        raise
                except httpx.TransportError:
                    if attempt >= self.max_retries:
                        raise
                time.sleep(min(2 ** attempt, 8))
            assert completion is not None
        if self.wire_api == "responses":
            return _parse_response_output(completion, schema)
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
    """Collect Responses API SSE deltas into a response-shaped dictionary."""
    headers = getattr(response, "headers", {})
    content_type = str(headers.get("content-type", ""))
    if "text/event-stream" not in content_type:
        return response.json()
    deltas: list[str] = []
    completed: dict | None = None
    for line in response.iter_lines():
        if not isinstance(line, str) or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            deltas.append(event["delta"])
            # Structured output is requested in the developer prompt.  Once a
            # complete JSON object is available, stop consuming trailing
            # reasoning/telemetry events and close the streamed response.
            candidate = "".join(deltas).strip()
            if candidate.startswith("{"):
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    return {"output_text": candidate}
        elif event_type == "response.completed":
            response_value = event.get("response")
            if isinstance(response_value, dict):
                completed = response_value
        elif event_type == "response.failed":
            response_value = event.get("response")
            completed = response_value if isinstance(response_value, dict) else event
    if completed is None:
        completed = {}
    if deltas:
        completed["output_text"] = "".join(deltas)
    return completed


def schema_instruction(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)
