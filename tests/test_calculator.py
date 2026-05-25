"""Calculator tests."""

import pytest

from tools.calculator import UnsafeExpressionError, evaluate_expression


def test_calculator_addition() -> None:
    assert evaluate_expression("2 + 2") == 4


def test_calculator_subtraction() -> None:
    assert evaluate_expression("10 - 3") == 7


def test_calculator_multiplication() -> None:
    assert evaluate_expression("17 * 23") == 391


def test_calculator_division() -> None:
    assert evaluate_expression("8 / 2") == 4


def test_calculator_power() -> None:
    assert evaluate_expression("2 ** 3") == 8
    assert evaluate_expression("2 ^ 3") == 8


def test_calculator_parentheses() -> None:
    assert evaluate_expression("(2 + 3) * 4") == 20


def test_calculator_modulo() -> None:
    assert evaluate_expression("10 % 3") == 1


def test_calculator_rejects_import() -> None:
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression("__import__('os')")


def test_calculator_rejects_dunder() -> None:
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression("__builtins__")


def test_calculator_rejects_letters() -> None:
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression("2 + abc")


def test_calculator_rejects_empty() -> None:
    with pytest.raises(UnsafeExpressionError):
        evaluate_expression("   ")


def test_calculator_division_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        evaluate_expression("1 / 0")
