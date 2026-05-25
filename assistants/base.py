"""Base assistant abstraction."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from assistants.memory import get_default_max_turns
from assistants.prompts import build_messages
from assistants.types import AssistantResult


class BaseAssistant(ABC):
    """Model inference layer; safety and tools are applied in pipeline.py."""

    assistant_type: str = "unknown"

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier for logging and results."""

    @abstractmethod
    def _generate_model_response(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Call the underlying model; implemented by OSS/Gemini subclasses."""

    def generate(
        self,
        message: str,
        history: list[dict[str, str]],
        **kwargs: Any,
    ) -> AssistantResult:
        """Run model inference for one user turn (no guardrails or tools)."""
        memory_turns = kwargs.get("memory_turns", get_default_max_turns())
        messages = build_messages(message, history, max_turns=memory_turns)

        start = time.perf_counter()
        try:
            text = self._generate_model_response(messages, **kwargs)
            error = None
        except Exception as exc:
            text = f"An error occurred while generating a response: {exc}"
            error = str(exc)

        elapsed = time.perf_counter() - start
        return AssistantResult(
            text=text,
            latency_seconds=elapsed,
            latency_model_seconds=elapsed,
            model_name=self.model_name,
            assistant_type=self.assistant_type,
            safety_triggered=False,
            tool_used=None,
            error=error,
        )
