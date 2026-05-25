"""Tests for Gemini assistant (API calls mocked)."""

from __future__ import annotations

import pytest

from assistants.gemini_assistant import DEFAULT_GEMINI_MODEL, GeminiAssistant


def test_gemini_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = GeminiAssistant().generate("Hello", [])
    assert result.error == "missing_api_key"
    assert result.latency_model_seconds is None
    assert "GEMINI_API_KEY" in result.text


def test_gemini_generate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    monkeypatch.setattr(
        "assistants.gemini_assistant._call_gemini",
        lambda messages, model_name: "Gemini reply",
    )

    result = GeminiAssistant().generate("Hi", [])
    assert result.error is None
    assert result.text == "Gemini reply"
    assert result.assistant_type == "gemini"
    assert result.latency_model_seconds is not None
    assert result.latency_model_seconds == result.latency_seconds
