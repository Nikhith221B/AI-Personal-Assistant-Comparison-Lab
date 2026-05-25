"""Rolling short-term conversation memory."""

from __future__ import annotations

import os

MIN_MEMORY_TURNS = 2
MAX_MEMORY_TURNS = 12
DEFAULT_MEMORY_TURNS = 8


def clamp_memory_turns(max_turns: int) -> int:
    """Clamp exchange count to the supported range (2–12)."""
    return max(MIN_MEMORY_TURNS, min(MAX_MEMORY_TURNS, max_turns))


def get_default_max_turns() -> int:
    """Read MAX_MEMORY_TURNS from the environment; default 8."""
    raw = os.getenv("MAX_MEMORY_TURNS", str(DEFAULT_MEMORY_TURNS))
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MEMORY_TURNS
    return clamp_memory_turns(value)


class ConversationMemory:
    """Keeps the last N exchanges (1 exchange = user + assistant pair)."""

    def __init__(self, max_turns: int | None = None) -> None:
        self.max_turns = clamp_memory_turns(
            max_turns if max_turns is not None else get_default_max_turns()
        )
        self._history: list[dict[str, str]] = []

    def add_exchange(self, user_message: str, assistant_message: str) -> None:
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": assistant_message})
        self._trim()

    def get_history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def set_max_turns(self, max_turns: int) -> None:
        self.max_turns = clamp_memory_turns(max_turns)
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    @staticmethod
    def trim_history(
        history: list[dict[str, str]],
        max_turns: int,
    ) -> list[dict[str, str]]:
        """Return the last N exchanges from a message list."""
        clamped = clamp_memory_turns(max_turns)
        max_messages = clamped * 2
        if len(history) <= max_messages:
            return list(history)
        return list(history[-max_messages:])
