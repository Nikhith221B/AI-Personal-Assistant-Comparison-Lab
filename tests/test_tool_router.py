"""Tests for tool routing."""

from tools.tool_router import route_tool


def test_route_calculator_what_is() -> None:
    result = route_tool("What is 17 * 23?")
    assert result is not None
    assert result.tool_name == "calculator"
    assert result.answer == "391"


def test_route_calculator_pure_expression() -> None:
    result = route_tool("(2 + 3) * 4")
    assert result is not None
    assert result.answer == "20"


def test_route_datetime_current_date() -> None:
    result = route_tool("What is the current date?")
    assert result is not None
    assert result.tool_name == "datetime"
    assert "current local date and time" in result.answer


def test_route_datetime_what_time() -> None:
    result = route_tool("What time is it?")
    assert result is not None
    assert result.tool_name == "datetime"


def test_route_none_for_general_chat() -> None:
    assert route_tool("Tell me a joke about programming.") is None
