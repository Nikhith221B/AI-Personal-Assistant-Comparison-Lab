"""Validate evaluation prompt set structure."""

import json
from collections import Counter
from pathlib import Path

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "evals" / "eval_prompts.json"
REQUIRED_COUNTS = {
    "factual": 10,
    "jailbreak": 10,
    "bias": 10,
    "memory": 5,
    "tool_use": 5,
}


def test_eval_prompts_json_valid() -> None:
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 40

    counts = Counter(item["category"] for item in data)
    for category, expected in REQUIRED_COUNTS.items():
        assert counts[category] == expected, f"{category}: expected {expected}, got {counts[category]}"

    ids = [item["id"] for item in data]
    assert len(ids) == len(set(ids)), "Duplicate prompt ids found"

    for item in data:
        assert "prompt" in item
        assert "conversation_prefix" in item
        assert isinstance(item["conversation_prefix"], list)
        if item["category"] == "memory":
            assert len(item["conversation_prefix"]) >= 2
        assert item.get("expected_behavior")
