"""Structured assistant response shared by pipeline, logs, and eval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssistantResult:
    text: str
    latency_seconds: float
    latency_model_seconds: float | None
    model_name: str
    assistant_type: str
    safety_triggered: bool
    tool_used: str | None
    error: str | None
    scoring_method: str | None = None
