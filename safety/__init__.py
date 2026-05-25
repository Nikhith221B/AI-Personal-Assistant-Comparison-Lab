"""Lightweight safety guardrails."""

from safety.guardrails import SafeCheckResult, check_prompt, check_response
from safety.refusal_templates import GENERIC_REFUSAL, get_refusal

__all__ = [
    "GENERIC_REFUSAL",
    "SafeCheckResult",
    "check_prompt",
    "check_response",
    "get_refusal",
]
