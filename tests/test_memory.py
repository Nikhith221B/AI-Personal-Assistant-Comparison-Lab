"""Tests for conversation memory and prompt formatting."""

import pytest

from assistants.memory import (
    MAX_MEMORY_TURNS,
    MIN_MEMORY_TURNS,
    ConversationMemory,
    clamp_memory_turns,
    get_default_max_turns,
)
from assistants.prompts import SYSTEM_PROMPT, build_messages, format_memory_context


def test_clamp_memory_turns_bounds() -> None:
    assert clamp_memory_turns(1) == MIN_MEMORY_TURNS
    assert clamp_memory_turns(99) == MAX_MEMORY_TURNS
    assert clamp_memory_turns(8) == 8


def test_get_default_max_turns_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MEMORY_TURNS", "10")
    assert get_default_max_turns() == 10
    monkeypatch.setenv("MAX_MEMORY_TURNS", "not-a-number")
    assert get_default_max_turns() == 8


def test_memory_trims_exchanges() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.add_exchange("u1", "a1")
    memory.add_exchange("u2", "a2")
    memory.add_exchange("u3", "a3")
    history = memory.get_history()
    assert history == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_memory_clear() -> None:
    memory = ConversationMemory(max_turns=4)
    memory.add_exchange("u", "a")
    memory.clear()
    assert memory.get_history() == []


def test_memory_set_max_turns_retrims() -> None:
    memory = ConversationMemory(max_turns=3)
    for i in range(4):
        memory.add_exchange(f"u{i}", f"a{i}")
    assert len(memory.get_history()) == 6
    memory.set_max_turns(MIN_MEMORY_TURNS)
    assert memory.get_history() == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_trim_history_static() -> None:
    full: list[dict[str, str]] = []
    for i in range(3):
        full.append({"role": "user", "content": f"u{i}"})
        full.append({"role": "assistant", "content": f"a{i}"})
    trimmed = ConversationMemory.trim_history(full, max_turns=2)
    assert len(trimmed) == 4
    assert trimmed[0]["content"] == "u1"


def test_format_memory_context_trims() -> None:
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"u{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    ctx = format_memory_context(history, max_turns=2)
    assert len(ctx) == 4
    assert ctx[0]["content"] == "u3"


def test_build_messages_structure() -> None:
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    messages = build_messages("What is 2+2?", history, max_turns=8)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1] == {"role": "user", "content": "What is 2+2?"}
    assert messages[1:-1] == history


def test_build_messages_respects_max_turns() -> None:
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"u{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    messages = build_messages("latest", history, max_turns=MIN_MEMORY_TURNS)
    assert len(messages) == 6  # system + 2 exchanges + current user
    assert messages[-1]["content"] == "latest"
    assert messages[1]["content"] == "u3"


def test_oss_generate_returns_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistants.oss_assistant import OSSAssistant

    monkeypatch.setattr("assistants.oss_assistant._ensure_loaded", lambda: None)
    monkeypatch.setattr("assistants.oss_assistant._generate_text", lambda _m: "Hello!")

    result = OSSAssistant().generate(
        "test",
        [{"role": "user", "content": "prior"}],
        memory_turns=8,
    )
    assert result.assistant_type == "oss"
    assert result.safety_triggered is False
    assert result.tool_used is None
    assert result.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert result.text == "Hello!"
    assert result.latency_model_seconds is not None
    assert result.error is None
