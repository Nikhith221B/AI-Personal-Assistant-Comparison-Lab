"""Tests for stratified evaluation prompt selection."""

from evals.run_eval import select_eval_prompts, _load_prompts


def test_select_all_when_max_none() -> None:
    prompts = _load_prompts()
    selected = select_eval_prompts(prompts, None)
    assert len(selected) == len(prompts)


def test_select_ten_is_stratified_not_head_factual_only() -> None:
    prompts = _load_prompts()
    selected = select_eval_prompts(prompts, 10)
    categories = {item["category"] for item in selected}
    assert categories == {"factual", "jailbreak", "bias", "memory", "tool_use"}
    assert len(selected) == 10
    assert all(item["category"] == "factual" for item in selected) is False


def test_select_ten_two_per_category() -> None:
    prompts = _load_prompts()
    selected = select_eval_prompts(prompts, 10)
    from collections import Counter

    counts = Counter(item["category"] for item in selected)
    assert counts["factual"] == 2
    assert counts["jailbreak"] == 2
    assert counts["bias"] == 2
    assert counts["memory"] == 2
    assert counts["tool_use"] == 2


def test_select_seven_distributes_remainder() -> None:
    prompts = _load_prompts()
    selected = select_eval_prompts(prompts, 7)
    from collections import Counter

    counts = Counter(item["category"] for item in selected)
    assert sum(counts.values()) == 7
    assert len(counts) == 5
    assert counts["factual"] == 2
    assert counts["jailbreak"] == 2
