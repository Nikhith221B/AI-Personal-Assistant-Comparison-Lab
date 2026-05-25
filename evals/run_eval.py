"""Evaluation runner for OSS vs Gemini comparison."""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = Path(__file__).resolve().parent / "eval_prompts.json"
RESULTS_CSV = ROOT / "reports" / "results.csv"
RESULTS_JSON = ROOT / "reports" / "results.json"
EVAL_LOG_PATH = ROOT / "logs" / "eval_runs.jsonl"

FIELDNAMES = [
    "prompt_id",
    "category",
    "assistant_type",
    "model_name",
    "prompt",
    "response",
    "latency_seconds",
    "latency_model_seconds",
    "safety_triggered",
    "tool_used",
    "error",
    "score",
    "pass_fail",
    "notes",
    "scoring_method",
]


def _load_prompts() -> list[dict]:
    if not PROMPTS_PATH.exists():
        return []
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def select_eval_prompts(
    prompts: list[dict],
    max_prompts: int | None = None,
) -> list[dict]:
    """Select prompts for an eval run.

    When ``max_prompts`` is set, take a balanced slice across categories
    (e.g. 10 prompts -> 2 per category for the default 5 categories) instead
    of the first N prompts in file order (which are all factual).
    """
    if not prompts:
        return []
    if max_prompts is None or max_prompts >= len(prompts):
        return list(prompts)

    max_prompts = max(1, max_prompts)

    by_category: dict[str, list[dict]] = {}
    category_order: list[str] = []
    for item in prompts:
        category = item.get("category", "unknown")
        if category not in by_category:
            by_category[category] = []
            category_order.append(category)
        by_category[category].append(item)

    n_categories = len(category_order)
    base = max_prompts // n_categories
    remainder = max_prompts % n_categories

    selected: list[dict] = []
    for index, category in enumerate(category_order):
        take = base + (1 if index < remainder else 0)
        if take > 0:
            selected.extend(by_category[category][:take])

    return selected


def _selection_summary(prompts: list[dict]) -> dict:
    from collections import Counter

    counts = Counter(item.get("category", "unknown") for item in prompts)
    return {
        "total_prompts": len(prompts),
        "per_category": dict(sorted(counts.items())),
    }


def _make_assistant(assistant_type: str):
    from assistants.gemini_assistant import GeminiAssistant
    from assistants.oss_assistant import OSSAssistant

    if assistant_type == "gemini":
        return GeminiAssistant()
    return OSSAssistant()


def _is_retryable_error(result) -> bool:
    if not result.error:
        return False
    err = str(result.error).lower()
    return "429" in err or "503" in err or "rate" in err or "quota" in err


def _run_turn_with_retry(assistant, user_message: str, history: list[dict]):
    from assistants.pipeline import run_turn

    last_result = None
    for attempt in range(2):
        if assistant.assistant_type == "gemini" and attempt == 0:
            time.sleep(1)
        last_result = run_turn(
            assistant,
            user_message,
            history,
            guardrails_enabled=True,
            tools_enabled=True,
            memory_turns=8,
        )
        if not _is_retryable_error(last_result):
            return last_result
        time.sleep(2)
    return last_result


def _save_results(rows: list[dict]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_eval_run(metrics: dict, total_rows: int) -> None:
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "metrics": metrics,
    }
    with EVAL_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def rescore_results(
    *,
    use_llm_judge: bool = False,
    results_path: Path | None = None,
) -> dict:
    """Re-score saved results with current judge rules (no model re-run)."""
    from evals.judge import score_row
    from evals.metrics import compute_metrics

    path = results_path or RESULTS_JSON
    if not path.exists():
        return {"status": "no_results", "message": f"Missing {path}"}

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list) or not rows:
        return {"status": "empty", "message": "No rows to rescore"}

    prompts_by_id = {item["id"]: item for item in _load_prompts()}
    updated: list[dict] = []

    for row in rows:
        prompt_id = row.get("prompt_id", "")
        prompt_item = prompts_by_id.get(prompt_id, {})
        if not prompt_item:
            prompt_item = {
                "id": prompt_id,
                "category": row.get("category", ""),
                "prompt": row.get("prompt", ""),
                "expected_answer": None,
                "expected_behavior": "",
            }

        scored = score_row(
            prompt_item,
            assistant_type=row.get("assistant_type", ""),
            response=row.get("response") or "",
            safety_triggered=bool(row.get("safety_triggered")),
            tool_used=row.get("tool_used") or None,
            error=row.get("error") or None,
            use_llm_judge=use_llm_judge,
        )
        updated.append(
            {
                **row,
                "score": scored.score,
                "pass_fail": scored.pass_fail,
                "notes": scored.notes,
                "scoring_method": scored.scoring_method,
            }
        )

    _save_results(updated)
    metrics = compute_metrics(updated)
    metrics["use_llm_judge"] = use_llm_judge
    metrics["rescored"] = True
    _log_eval_run(metrics, len(updated))

    return {
        "status": "ok",
        "rows": updated,
        "metrics": metrics,
        "message": f"Rescored {len(updated)} rows (use_llm_judge={use_llm_judge}).",
    }


def run_evaluation(
    max_prompts: int | None = None,
    assistants: list[str] | None = None,
    use_llm_judge: bool | None = None,
) -> dict:
    """Run prompts through the shared pipeline, score, and save results."""
    from evals.judge import score_row
    from evals.metrics import compute_metrics

    assistant_types = assistants or ["oss", "gemini"]
    all_prompts = _load_prompts()
    prompts = select_eval_prompts(all_prompts, max_prompts)

    if use_llm_judge is None:
        # Subset runs skip extra Gemini judge calls unless explicitly enabled.
        use_llm_judge = max_prompts is None

    selection = _selection_summary(prompts)

    if not prompts:
        return {
            "results": [],
            "metrics": {"status": "no_prompts"},
            "message": "No evaluation prompts found in evals/eval_prompts.json.",
            "rows": [],
        }

    rows: list[dict] = []
    gemini_quota_exhausted = False

    for item in prompts:
        prompt_id = item.get("id", "unknown")
        category = item.get("category", "unknown")
        user_message = item.get("prompt", "")
        history = item.get("conversation_prefix", []) or []

        for assistant_type in assistant_types:
            assistant = _make_assistant(assistant_type)

            if assistant_type == "gemini" and gemini_quota_exhausted:
                from assistants.types import AssistantResult

                result = AssistantResult(
                    text="Gemini quota exhausted for this eval run; skipped remaining Gemini calls.",
                    latency_seconds=0.0,
                    latency_model_seconds=None,
                    model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                    assistant_type="gemini",
                    safety_triggered=False,
                    tool_used=None,
                    error="gemini_quota_exhausted",
                )
            elif assistant_type == "gemini" and not os.getenv("GEMINI_API_KEY"):
                result = assistant.generate(user_message, history, memory_turns=8)
            else:
                result = _run_turn_with_retry(assistant, user_message, history)

            if assistant_type == "gemini" and _is_retryable_error(result):
                gemini_quota_exhausted = True

            scored = score_row(
                item,
                assistant_type=result.assistant_type,
                response=result.text,
                safety_triggered=result.safety_triggered,
                tool_used=result.tool_used,
                error=result.error,
                use_llm_judge=use_llm_judge,
            )

            rows.append(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "assistant_type": result.assistant_type,
                    "model_name": result.model_name,
                    "prompt": user_message,
                    "response": result.text,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "latency_model_seconds": (
                        round(result.latency_model_seconds, 3)
                        if result.latency_model_seconds is not None
                        else ""
                    ),
                    "safety_triggered": result.safety_triggered,
                    "tool_used": result.tool_used or "",
                    "error": result.error or "",
                    "score": scored.score,
                    "pass_fail": scored.pass_fail,
                    "notes": scored.notes,
                    "scoring_method": scored.scoring_method,
                }
            )

    _save_results(rows)
    metrics = compute_metrics(rows)
    metrics["use_llm_judge"] = use_llm_judge
    metrics["prompt_selection"] = {
        "max_prompts": max_prompts,
        "available_prompts": len(all_prompts),
        **selection,
    }
    _log_eval_run(metrics, len(rows))

    selection_note = (
        f"Selected {selection['total_prompts']} prompts (stratified): "
        f"{selection['per_category']}."
    )
    if max_prompts is None:
        selection_note = f"Running all {selection['total_prompts']} prompts."

    return {
        "results": rows,
        "metrics": metrics,
        "message": selection_note,
        "rows": rows,
    }


def _print_metric_summary(metrics: dict) -> None:
    print("\n=== Evaluation Metrics ===")
    skip_keys = {"use_llm_judge", "prompt_selection"}
    for assistant_type, values in metrics.items():
        if assistant_type in skip_keys or not isinstance(values, dict):
            continue
        print(f"\n[{assistant_type}] {values.get('model_name', '')}")
        for key, value in values.items():
            if key in {"model_name", "disclaimer"}:
                continue
            print(f"  {key}: {value}")
        if values.get("disclaimer"):
            print(f"  note: {values['disclaimer']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OSS vs Gemini evaluation")
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limit prompts with balanced per-category sampling "
            "(default: all 40). Example: 10 -> 2 per category."
        ),
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Use rule-based scoring only (saves Gemini quota)",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Re-score reports/results.json with current judge rules (no model calls)",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="With --rescore-only, also re-run Gemini LLM judge on OSS factual/memory rows",
    )
    args = parser.parse_args()
    use_llm_judge = False if args.no_llm_judge else None

    if args.rescore_only:
        summary = rescore_results(use_llm_judge=args.with_llm_judge)
    else:
        summary = run_evaluation(max_prompts=args.max_prompts, use_llm_judge=use_llm_judge)
    print(json.dumps(summary["metrics"], indent=2))
    _print_metric_summary(summary["metrics"])
    if summary.get("message"):
        print(summary["message"])
    print(f"\nSaved: {RESULTS_CSV} and {RESULTS_JSON}")

    from evals.generate_report import generate_report

    generate_report()
    print(f"Report: {ROOT / 'reports' / 'evaluation_report.md'}")
