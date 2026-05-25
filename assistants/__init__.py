"""Assistant modules for OSS and Gemini models."""

from dotenv import load_dotenv

load_dotenv()

from assistants.base import BaseAssistant
from assistants.memory import (
    ConversationMemory,
    clamp_memory_turns,
    get_default_max_turns,
)
from assistants.prompts import SYSTEM_PROMPT, build_messages, format_memory_context
from assistants.types import AssistantResult

__all__ = [
    "AssistantResult",
    "BaseAssistant",
    "ConversationMemory",
    "SYSTEM_PROMPT",
    "build_messages",
    "clamp_memory_turns",
    "format_memory_context",
    "get_default_max_turns",
]
