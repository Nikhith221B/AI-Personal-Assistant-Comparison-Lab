"""Aggregate metrics from evaluation results."""

from __future__ import annotations

import os
from collections import defaultdict

# Documented defaults when token usage metadata is unavailable (Gemini 2.5 Flash, USD per 1M tokens).
# Verify against current Google AI pricing: https://ai.google.dev/pricing
GEMINI_PRICE_PER_1M = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "default": {"input": 0.10, "output": 0.40},
}

DEFAULT_INPUT_TOKENS = 450
DEFAULT_OUTPUT_TOKENS = 180

METRIC_DISCLAIMER = (
    "Directional demo metrics only (small per-category sample sizes); "
    "not statistically significant."
)


def _rate(passed: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(passed / total, 4)


def _rows_for(rows: list[dict], assistant_type: str) -> list[dict]:
    return [row for row in rows if row.get("assistant_type") == assistant_type]


def _passed(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("pass_fail") == "pass")


def estimate_prompt_cost_usd(
    *,
    assistant_type: str,
    model_name: str,
    prompt: str,
    response: str,
) -> float:
    """Estimate per-prompt cost. OSS local inference is $0."""
    if assistant_type != "gemini":
        return 0.0

    pricing = GEMINI_PRICE_PER_1M.get(model_name, GEMINI_PRICE_PER_1M["default"])
    input_tokens = max(DEFAULT_INPUT_TOKENS, len(prompt) // 4 + 120)
    output_tokens = max(20, len(response) // 4)
    cost = (
        input_tokens * pricing["input"] + output_tokens * pricing["output"]
    ) / 1_000_000
    return round(cost, 8)


def compute_metrics(rows: list[dict]) -> dict:
    """Compute per-assistant metrics from scored result rows."""
    metrics: dict[str, dict] = {}
    assistant_types = sorted({row.get("assistant_type", "unknown") for row in rows})

    for assistant_type in assistant_types:
        subset = _rows_for(rows, assistant_type)
        if not subset:
            continue

        by_category: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            by_category[row.get("category", "unknown")].append(row)

        factual = by_category.get("factual", [])
        jailbreak = by_category.get("jailbreak", [])
        bias = by_category.get("bias", [])
        memory = by_category.get("memory", [])
        tool_use = by_category.get("tool_use", [])

        jailbreak_passed = _passed(jailbreak)
        bias_passed = _passed(bias)
        safety_total = len(jailbreak) + len(bias)
        safety_passed = jailbreak_passed + bias_passed

        latencies = [float(row["latency_seconds"]) for row in subset if row.get("latency_seconds") != ""]
        model_latencies = [
            float(row["latency_model_seconds"])
            for row in subset
            if row.get("latency_model_seconds") not in ("", None)
        ]

        costs = [
            estimate_prompt_cost_usd(
                assistant_type=assistant_type,
                model_name=row.get("model_name", ""),
                prompt=row.get("prompt", ""),
                response=row.get("response", ""),
            )
            for row in subset
        ]
        mean_cost = sum(costs) / len(costs) if costs else 0.0

        factual_pass_rate = _rate(_passed(factual), len(factual))
        metrics[assistant_type] = {
            "factual_pass_rate": factual_pass_rate,
            # Backward compatibility for older eval_runs.jsonl entries and PRD naming.
            "hallucination_rate": factual_pass_rate,
            "jailbreak_resistance_rate": _rate(jailbreak_passed, len(jailbreak)),
            "bias_safety_pass_rate": _rate(bias_passed, len(bias)),
            "content_safety_pass_rate": _rate(safety_passed, safety_total),
            "memory_consistency_rate": _rate(_passed(memory), len(memory)),
            "tool_use_success_rate": _rate(_passed(tool_use), len(tool_use)),
            "average_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "average_latency_model_seconds": (
                round(sum(model_latencies) / len(model_latencies), 4) if model_latencies else None
            ),
            "estimated_cost_per_100_prompts": round(mean_cost * 100, 6),
            "total_rows": len(subset),
            "model_name": subset[0].get("model_name"),
            "disclaimer": METRIC_DISCLAIMER,
        }

    return metrics
