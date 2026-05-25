"""Gemini frontier assistant via Google GenAI Python SDK."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from google import genai
from google.genai import types

from assistants.base import BaseAssistant
from assistants.types import AssistantResult

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None
_executor = ThreadPoolExecutor(max_workers=2)


def _request_timeout_seconds() -> float:
    raw = os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "60")
    try:
        return float(raw)
    except ValueError:
        return 60.0


def _get_client() -> genai.Client:
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if _client is None:
        timeout_ms = int(_request_timeout_seconds() * 1000)
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
    return _client


def _messages_to_gemini(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[types.Content]]:
    """Split system prompt and convert chat history to Gemini contents."""
    system_instruction: str | None = None
    contents: list[types.Content] = []

    for message in messages:
        role = message.get("role", "")
        text = message.get("content", "")
        if role == "system":
            system_instruction = text
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))

    return system_instruction, contents


def _call_gemini(messages: list[dict[str, str]], model_name: str) -> str:
    client = _get_client()
    system_instruction, contents = _messages_to_gemini(messages)

    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = (response.text or "").strip()
    return text or "(No response generated.)"


class GeminiAssistant(BaseAssistant):
    assistant_type = "gemini"

    @property
    def model_name(self) -> str:
        return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def generate(
        self,
        message: str,
        history: list[dict[str, str]],
        **kwargs: Any,
    ) -> AssistantResult:
        if not os.getenv("GEMINI_API_KEY"):
            return AssistantResult(
                text="Gemini is not configured. Set GEMINI_API_KEY in your .env file.",
                latency_seconds=0.0,
                latency_model_seconds=None,
                model_name=self.model_name,
                assistant_type=self.assistant_type,
                safety_triggered=False,
                tool_used=None,
                error="missing_api_key",
            )
        return super().generate(message, history, **kwargs)

    def _generate_model_response(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del kwargs
        model_name = self.model_name
        timeout = _request_timeout_seconds()
        future = _executor.submit(_call_gemini, messages, model_name)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Gemini request exceeded {timeout:.0f}s. "
                "Try again or increase GEMINI_REQUEST_TIMEOUT_SECONDS."
            ) from exc
