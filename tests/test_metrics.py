"""Tests for evaluation metrics aggregation."""

from evals.metrics import compute_metrics, estimate_prompt_cost_usd


def test_estimate_oss_cost_zero() -> None:
    assert estimate_prompt_cost_usd(
        assistant_type="oss",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        prompt="hi",
        response="hello",
    ) == 0.0


def test_compute_metrics_rates() -> None:
    rows = [
        {
            "assistant_type": "oss",
            "model_name": "qwen",
            "category": "factual",
            "pass_fail": "pass",
            "latency_seconds": 1.0,
            "latency_model_seconds": 1.0,
            "prompt": "q",
            "response": "a",
        },
        {
            "assistant_type": "oss",
            "model_name": "qwen",
            "category": "factual",
            "pass_fail": "fail",
            "latency_seconds": 2.0,
            "latency_model_seconds": 2.0,
            "prompt": "q",
            "response": "b",
        },
        {
            "assistant_type": "oss",
            "model_name": "qwen",
            "category": "jailbreak",
            "pass_fail": "pass",
            "latency_seconds": 0.5,
            "latency_model_seconds": 0.5,
            "prompt": "q",
            "response": "refuse",
        },
    ]
    metrics = compute_metrics(rows)
    assert metrics["oss"]["factual_pass_rate"] == 0.5
    assert metrics["oss"]["hallucination_rate"] == 0.5
    assert metrics["oss"]["jailbreak_resistance_rate"] == 1.0
    assert metrics["oss"]["estimated_cost_per_100_prompts"] == 0.0
