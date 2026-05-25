"""Shared request pipeline: guardrails -> tools -> model -> post-guardrail."""

from __future__ import annotations

import time
from typing import Any

from assistants.base import BaseAssistant
from assistants.oss_assistant import OSSAssistant, get_load_error, is_model_loaded, is_model_loading
from assistants.types import AssistantResult
from safety.guardrails import check_prompt, check_response
from safety.refusal_templates import get_refusal
from tools.tool_router import route_tool


def run_turn(
    assistant: BaseAssistant,
    message: str,
    history: list[dict[str, str]],
    *,
    guardrails_enabled: bool = True,
    tools_enabled: bool = True,
    memory_turns: int = 8,
    **kwargs: Any,
) -> AssistantResult:
    """Execute one assistant turn through the full pipeline."""
    start = time.perf_counter()
    message = (message or "").strip()
    if not message:
        elapsed = time.perf_counter() - start
        return AssistantResult(
            text="Please enter a message.",
            latency_seconds=elapsed,
            latency_model_seconds=None,
            model_name=assistant.model_name,
            assistant_type=assistant.assistant_type,
            safety_triggered=False,
            tool_used=None,
            error="empty_message",
        )

    if isinstance(assistant, OSSAssistant):
        if is_model_loading():
            elapsed = time.perf_counter() - start
            return AssistantResult(
                text="Loading open-source model… Please try again in a moment.",
                latency_seconds=elapsed,
                latency_model_seconds=None,
                model_name=assistant.model_name,
                assistant_type=assistant.assistant_type,
                safety_triggered=False,
                tool_used=None,
                error="model_loading",
            )
        load_err = get_load_error()
        if load_err and not is_model_loaded():
            elapsed = time.perf_counter() - start
            return AssistantResult(
                text=f"Failed to load OSS model: {load_err}",
                latency_seconds=elapsed,
                latency_model_seconds=None,
                model_name=assistant.model_name,
                assistant_type=assistant.assistant_type,
                safety_triggered=False,
                tool_used=None,
                error="model_load_failed",
            )

    if guardrails_enabled:
        pre_check = check_prompt(message)
        if not pre_check.is_safe:
            elapsed = time.perf_counter() - start
            return AssistantResult(
                text=get_refusal(pre_check.category),
                latency_seconds=elapsed,
                latency_model_seconds=None,
                model_name=assistant.model_name,
                assistant_type=assistant.assistant_type,
                safety_triggered=True,
                tool_used=None,
                error=None,
            )

    if tools_enabled:
        tool_result = route_tool(message)
        if tool_result is not None:
            elapsed = time.perf_counter() - start
            return AssistantResult(
                text=tool_result.answer,
                latency_seconds=elapsed,
                latency_model_seconds=None,
                model_name=assistant.model_name,
                assistant_type=assistant.assistant_type,
                safety_triggered=False,
                tool_used=tool_result.tool_name,
                error=None,
            )

    result = assistant.generate(
        message,
        history,
        memory_turns=memory_turns,
        **kwargs,
    )

    if guardrails_enabled and result.error is None and result.text:
        post_check = check_response(result.text)
        if not post_check.is_safe:
            result = AssistantResult(
                text=get_refusal(post_check.category),
                latency_seconds=result.latency_seconds,
                latency_model_seconds=result.latency_model_seconds,
                model_name=result.model_name,
                assistant_type=result.assistant_type,
                safety_triggered=True,
                tool_used=result.tool_used,
                error=result.error,
                scoring_method=result.scoring_method,
            )

    return result
