"""AST-based safe arithmetic evaluator."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

_ALLOWED_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains disallowed syntax."""


def _normalize_expression(expression: str) -> str:
    """Strip whitespace and allow ^ as power operator."""
    text = expression.strip()
    if not text:
        raise UnsafeExpressionError("Expression is empty.")
    return re.sub(r"\^", "**", text)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError("Only numeric constants are allowed.")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise UnsafeExpressionError(f"Operator {op_type.__name__} is not allowed.")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero.")
        return float(_ALLOWED_BINOPS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise UnsafeExpressionError(f"Unary operator {op_type.__name__} is not allowed.")
        return float(_ALLOWED_UNARYOPS[op_type](_eval_node(node.operand)))

    raise UnsafeExpressionError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely (+, -, *, /, //, %, **, parentheses)."""
    normalized = _normalize_expression(expression)
    if re.search(r"[a-zA-Z_]", normalized):
        raise UnsafeExpressionError("Letters and identifiers are not allowed.")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression: {exc}") from exc

    if not isinstance(tree.body, ast.AST):
        raise UnsafeExpressionError("Invalid expression.")

    result = _eval_node(tree.body)
    return float(result)
