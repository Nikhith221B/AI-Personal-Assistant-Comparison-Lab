"""Shared system prompt and message formatting."""

from __future__ import annotations

from assistants.memory import ConversationMemory, clamp_memory_turns

SYSTEM_PROMPT = (
    "You are a helpful, honest personal assistant. "
    "Answer clearly, refuse unsafe requests, and avoid stereotypes or harmful content."
)


def format_memory_context(
    history: list[dict[str, str]],
    max_turns: int,
) -> list[dict[str, str]]:
    """Return trimmed conversation history (no system prompt, no current user message)."""
    return ConversationMemory.trim_history(history, max_turns)


def build_messages(
    user_message: str,
    history: list[dict[str, str]],
    max_turns: int = 8,
) -> list[dict[str, str]]:
    """Build chat messages: system prompt + rolling memory + current user turn."""
    trimmed = format_memory_context(history, clamp_memory_turns(max_turns))
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_message})
    return messages
