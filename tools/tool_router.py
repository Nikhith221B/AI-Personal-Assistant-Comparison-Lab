"""Rule-based routing for calculator and datetime tools."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tools.calculator import UnsafeExpressionError, evaluate_expression
from tools.datetime_tool import current_datetime_iso

_MATH_PREFIX = re.compile(
    r"^(?:what(?:'s| is)|calculate|compute|evaluate|solve)\s+(.+?)\s*\??$",
    re.IGNORECASE,
)
_PURE_MATH = re.compile(r"^[\d\s+\-*/().%^]+$")
_HAS_OPERATOR = re.compile(r"[\+\-\*/\^%]")

_DATETIME_PATTERNS = (
    re.compile(r"\bwhat\s+time\s+is\s+it\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:current\s+)?(?:date|time|datetime)\b", re.IGNORECASE),
    re.compile(r"\b(?:current|today'?s?)\s+(?:date|time|datetime)\b", re.IGNORECASE),
    re.compile(r"\btoday'?s?\s+date\b", re.IGNORECASE),
)


@dataclass
class ToolResult:
    answer: str
    tool_name: str


def _extract_math_expression(user_message: str) -> str | None:
    text = user_message.strip()
    if not text:
        return None

    prefix_match = _MATH_PREFIX.match(text)
    if prefix_match:
        return prefix_match.group(1).strip()

    if _PURE_MATH.match(text) and _HAS_OPERATOR.search(text):
        return text

    return None


def _try_calculator(user_message: str) -> ToolResult | None:
    expression = _extract_math_expression(user_message)
    if not expression:
        return None

    try:
        value = evaluate_expression(expression)
    except ZeroDivisionError:
        return ToolResult(answer="Cannot divide by zero.", tool_name="calculator")
    except (UnsafeExpressionError, ValueError) as exc:
        return None

    if value == int(value):
        formatted = str(int(value))
    else:
        formatted = str(round(value, 10)).rstrip("0").rstrip(".")
    return ToolResult(answer=formatted, tool_name="calculator")


def _try_datetime(user_message: str) -> ToolResult | None:
    text = user_message.strip()
    for pattern in _DATETIME_PATTERNS:
        if pattern.search(text):
            return ToolResult(
                answer=f"The current local date and time is {current_datetime_iso()}.",
                tool_name="datetime",
            )
    return None


def route_tool(user_message: str) -> ToolResult | None:
    """Return a tool result if the message clearly requests calculator or datetime."""
    if not user_message or not user_message.strip():
        return None

    datetime_result = _try_datetime(user_message)
    if datetime_result is not None:
        return datetime_result

    return _try_calculator(user_message)
