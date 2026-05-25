"""Generate reports/evaluation_report.md from evaluation results."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from evals.metrics import METRIC_DISCLAIMER, compute_metrics

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "reports" / "results.csv"
RESULTS_JSON = ROOT / "reports" / "results.json"
EVAL_LOG_PATH = ROOT / "logs" / "eval_runs.jsonl"
REPORT_PATH = ROOT / "reports" / "evaluation_report.md"
CHARTS_DIR = ROOT / "reports" / "charts"

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


def _load_rows() -> tuple[list[dict], str]:
    """Load result rows and a short source label."""
    if RESULTS_JSON.exists():
        payload = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        generated = payload.get("generated_at", "unknown")
        return rows, f"reports/results.json ({generated})"

    if RESULTS_CSV.exists():
        with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows, "reports/results.csv"

    return [], ""


def _load_metrics_from_eval_log() -> tuple[dict, int, str] | None:
    """Fallback: latest eval_runs.jsonl entry with both assistants."""
    if not EVAL_LOG_PATH.exists():
        return None

    best: dict | None = None
    best_rows = 0
    best_ts = ""

    for line in EVAL_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        metrics = record.get("metrics", {})
        if "oss" in metrics and "gemini" in metrics:
            total = int(record.get("total_rows", 0))
            if total >= best_rows:
                best = metrics
                best_rows = total
                best_ts = record.get("timestamp", "")

    if best is None:
        return None
    return best, best_rows, best_ts


def _metric_bar(rate: float | None, *, invert: bool = False) -> str:
    if rate is None:
        return "— (not in sample)"
    display = 1.0 - rate if invert else rate
    filled = max(0, min(10, int(round(display * 10))))
    empty = 10 - filled
    pct = display * 100
    return f"{'█' * filled}{'░' * empty} {pct:.0f}%"


def _fmt_rate(rate: float | None) -> str:
    if rate is None:
        return "—"
    return f"{rate * 100:.1f}%"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}s"


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.4f}"


def _factual_pass_rate(metrics: dict) -> float | None:
    """Read factual pass rate from new or legacy metric keys."""
    if "factual_pass_rate" in metrics:
        return metrics.get("factual_pass_rate")
    return metrics.get("hallucination_rate")


def _oss_factual_failures(rows: list[dict], limit: int = 5) -> list[str]:
    """Summarize OSS factual rows that failed scoring."""
    lines: list[str] = []
    for row in rows:
        if row.get("assistant_type") != "oss" or row.get("category") != "factual":
            continue
        if row.get("pass_fail") == "pass":
            continue
        prompt_id = row.get("prompt_id", "?")
        response = (row.get("response") or "").replace("\n", " ").strip()
        if len(response) > 120:
            response = response[:117] + "..."
        error = row.get("error") or ""
        notes = row.get("notes") or ""
        detail = error or notes or "scoring failed"
        lines.append(f"`{prompt_id}`: {detail} — response: \"{response or '(empty)'}\"")
        if len(lines) >= limit:
            break
    return lines


def _categories_in_sample(rows: list[dict]) -> list[str]:
    cats = sorted({row.get("category", "unknown") for row in rows})
    return cats


def _error_summary(rows: list[dict], assistant_type: str) -> str:
    subset = [r for r in rows if r.get("assistant_type") == assistant_type]
    errors = [r.get("error") for r in subset if r.get("error")]
    if not errors:
        return "No errors recorded."
    unique = sorted(set(errors))
    return "; ".join(unique[:5]) + (f" (+{len(unique) - 5} more)" if len(unique) > 5 else "")


def _try_write_charts(metrics: dict) -> list[str]:
    """Optional matplotlib charts; returns relative paths written."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    labels = []
    factual = []
    latencies = []

    for key in ("oss", "gemini"):
        if key not in metrics:
            continue
        m = metrics[key]
        labels.append(key.upper())
        rate = _factual_pass_rate(m)
        factual.append((rate or 0) * 100)
        latencies.append(m.get("average_latency_seconds") or 0)

    if not labels:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].bar(labels, factual, color=["#4a90d9", "#e67e22"])
    axes[0].set_title("Factual pass rate (%)")
    axes[0].set_ylim(0, 100)

    axes[1].bar(labels, latencies, color=["#4a90d9", "#e67e22"])
    axes[1].set_title("Avg end-to-end latency (s)")
    fig.tight_layout()
    path = CHARTS_DIR / "comparison_snapshot.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append("reports/charts/comparison_snapshot.png")
    return written


def generate_report(*, write_charts: bool = True) -> str:
    """Build evaluation_report.md; returns report text."""
    rows, source = _load_rows()
    data_pending = not rows

    metrics: dict
    sample_note = ""

    if rows:
        metrics = compute_metrics(rows)
        prompt_ids = {r.get("prompt_id") for r in rows}
        n_prompts = len(prompt_ids) if prompt_ids else len(rows) // max(1, len(metrics))
        cats = _categories_in_sample(rows)
        per_cat = (metrics.get("prompt_selection") or {}).get("per_category")
        cat_detail = (
            f" per category: {per_cat}" if per_cat else f": {', '.join(cats) or 'unknown'}"
        )
        sample_note = (
            f"{n_prompts} prompts × {len(metrics)} assistants "
            f"({len(rows)} rows); categories{cat_detail}."
        )
    else:
        fallback = _load_metrics_from_eval_log()
        if fallback is None:
            pending = (
                "# Evaluation Report\n\n"
                "**Status:** Pending evaluation run.\n\n"
                "No `reports/results.csv` / `reports/results.json` found and no "
                "logged eval with both OSS and Gemini.\n\n"
                "Run:\n\n"
                "```bash\n"
                "python evals/run_eval.py\n"
                "```\n\n"
                "Then:\n\n"
                "```bash\n"
                "python evals/generate_report.py\n"
                "```\n\n"
                "For setup, architecture, deployment, and tradeoffs, see [README.md](../README.md).\n"
            )
            REPORT_PATH.write_text(pending, encoding="utf-8")
            return pending

        metrics, total_rows, ts = fallback
        sample_note = (
            f"Aggregates from latest `logs/eval_runs.jsonl` entry ({ts}, {total_rows} rows). "
            "Re-run eval and regenerate for row-level detail."
        )
        data_pending = True

    oss = metrics.get("oss", {})
    gemini = metrics.get("gemini", {})
    oss_name = oss.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    gemini_name = gemini.get("model_name", "gemini-2.5-flash")

    chart_paths = _try_write_charts(metrics) if write_charts else []

    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "For setup, architecture, deployment, and tradeoffs, see [README.md](../README.md).",
        "",
        "## Executive summary",
        "",
        "This lab compared **local OSS** (`" + oss_name + "`) against **frontier API** (`"
        + gemini_name
        + "`) on the same prompts through the shared pipeline (guardrails, tools, memory).",
        "",
    ]

    if data_pending and not rows:
        lines.append(
            "> **Data source:** Logged eval aggregates only — export files missing. "
            "Numbers below are from the last successful comparison run."
        )
        lines.append("")
    elif source:
        lines.append(f"> **Data source:** {source}")
        lines.append("")

    lines.extend(
        [
            f"**Sample:** {sample_note}",
            "",
            "**Headline (directional):**",
            f"- Factual pass rate: OSS {_fmt_rate(_factual_pass_rate(oss))} vs Gemini "
            f"{_fmt_rate(_factual_pass_rate(gemini))}.",
            f"- End-to-end latency: OSS {_fmt_seconds(oss.get('average_latency_seconds'))} vs Gemini "
            f"{_fmt_seconds(gemini.get('average_latency_seconds'))}.",
            f"- Est. cost / 100 prompts: OSS {_fmt_cost(oss.get('estimated_cost_per_100_prompts'))} vs Gemini "
            f"{_fmt_cost(gemini.get('estimated_cost_per_100_prompts'))}.",
            "",
            f"**Disclaimer:** {METRIC_DISCLAIMER}",
            "",
            "## Model comparison",
            "",
            "| Metric | OSS | Gemini |",
            "|--------|-----|--------|",
            f"| Factual pass rate | {_fmt_rate(_factual_pass_rate(oss))} | "
            f"{_fmt_rate(_factual_pass_rate(gemini))} |",
            f"| Jailbreak resistance | {_fmt_rate(oss.get('jailbreak_resistance_rate'))} | "
            f"{_fmt_rate(gemini.get('jailbreak_resistance_rate'))} |",
            f"| Bias safety pass | {_fmt_rate(oss.get('bias_safety_pass_rate'))} | "
            f"{_fmt_rate(gemini.get('bias_safety_pass_rate'))} |",
            f"| Content safety (combined) | {_fmt_rate(oss.get('content_safety_pass_rate'))} | "
            f"{_fmt_rate(gemini.get('content_safety_pass_rate'))} |",
            f"| Memory consistency | {_fmt_rate(oss.get('memory_consistency_rate'))} | "
            f"{_fmt_rate(gemini.get('memory_consistency_rate'))} |",
            f"| Tool-use success | {_fmt_rate(oss.get('tool_use_success_rate'))} | "
            f"{_fmt_rate(gemini.get('tool_use_success_rate'))} |",
            f"| Rows scored | {oss.get('total_rows', '—')} | {gemini.get('total_rows', '—')} |",
            "",
            "## Metric bars (higher bar = better on this chart)",
            "",
            "| Dimension | OSS | Gemini |",
            "|-----------|-----|--------|",
            f"| Factual accuracy | {_metric_bar(_factual_pass_rate(oss))} | "
            f"{_metric_bar(_factual_pass_rate(gemini))} |",
            f"| Jailbreak resistance | {_metric_bar(oss.get('jailbreak_resistance_rate'))} | "
            f"{_metric_bar(gemini.get('jailbreak_resistance_rate'))} |",
            f"| Bias safety | {_metric_bar(oss.get('bias_safety_pass_rate'))} | "
            f"{_metric_bar(gemini.get('bias_safety_pass_rate'))} |",
            f"| Memory consistency | {_metric_bar(oss.get('memory_consistency_rate'))} | "
            f"{_metric_bar(gemini.get('memory_consistency_rate'))} |",
            f"| Tool use | {_metric_bar(oss.get('tool_use_success_rate'))} | "
            f"{_metric_bar(gemini.get('tool_use_success_rate'))} |",
            "",
            "## Cost and latency",
            "",
            "| Assistant | End-to-end latency | Model-only latency | Est. cost / 100 prompts |",
            "|-----------|-------------------|--------------------|-------------------------|",
            f"| OSS | {_fmt_seconds(oss.get('average_latency_seconds'))} | "
            f"{_fmt_seconds(oss.get('average_latency_model_seconds'))} | "
            f"{_fmt_cost(oss.get('estimated_cost_per_100_prompts'))} |",
            f"| Gemini | {_fmt_seconds(gemini.get('average_latency_seconds'))} | "
            f"{_fmt_seconds(gemini.get('average_latency_model_seconds'))} | "
            f"{_fmt_cost(gemini.get('estimated_cost_per_100_prompts'))} |",
            "",
            "*OSS cost is $0 API; excludes local hardware/electricity. Gemini cost uses token estimates in `evals/metrics.py`.*",
            "",
        ]
    )

    if rows:
        lines.extend(
            [
                "## Run health",
                "",
                f"- **OSS errors:** {_error_summary(rows, 'oss')}",
                f"- **Gemini errors:** {_error_summary(rows, 'gemini')}",
                "",
            ]
        )
        oss_failures = _oss_factual_failures(rows)
        if oss_failures:
            lines.extend(
                [
                    "## OSS factual failures (sample)",
                    "",
                    "OSS factual rows use Gemini as an optional LLM judge when enabled; "
                    "if the judge returns no verdict or errors, scoring falls back to the same "
                    "fuzzy_match rules as Gemini (expected answer need not match exact wording).",
                    "",
                ]
            )
            for item in oss_failures:
                lines.append(f"- {item}")
            lines.append("")

    lines.extend(
        [
            "## Key findings",
            "",
        ]
    )

    findings = _derive_findings(metrics, rows)
    for item in findings:
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(
        [
            "## Recommendation",
            "",
            "- **Prefer OSS (Qwen 0.5B)** for zero marginal API cost, offline/local demos, and Hugging Face Spaces "
            "where only the small model is deployed. Expect weaker reasoning, safety, and memory on hard prompts.",
            "- **Prefer Gemini** when answer quality, safety refusals, and memory/tool reliability matter more than "
            "cost and latency — at the price of API quotas and per-token spend.",
            "- **Use this repo’s pipeline for both** so comparisons reflect guardrails and tools, not raw model-only behavior.",
            "",
        ]
    )

    if chart_paths:
        lines.append("## Charts")
        lines.append("")
        for path in chart_paths:
            rel = path.removeprefix("reports/").lstrip("/")
            lines.append(f"![Comparison snapshot]({rel})")
        lines.append("")

    text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def _derive_findings(metrics: dict, rows: list[dict]) -> list[str]:
    findings: list[str] = []
    oss = metrics.get("oss", {})
    gemini = metrics.get("gemini", {})

    o_fact = _factual_pass_rate(oss)
    g_fact = _factual_pass_rate(gemini)
    if o_fact is not None and g_fact is not None:
        if o_fact > g_fact:
            findings.append(
                f"On factual prompts in this sample, OSS passed more often ({_fmt_rate(o_fact)} vs {_fmt_rate(g_fact)})."
            )
        elif g_fact > o_fact:
            findings.append(
                f"On factual prompts in this sample, Gemini passed more often ({_fmt_rate(g_fact)} vs {_fmt_rate(o_fact)})."
            )
        else:
            findings.append(f"Factual pass rates tied at {_fmt_rate(o_fact)} in this sample.")

    o_lat = oss.get("average_latency_seconds")
    g_lat = gemini.get("average_latency_seconds")
    if o_lat is not None and g_lat is not None:
        faster = "OSS" if o_lat < g_lat else "Gemini" if g_lat < o_lat else "Neither"
        if faster != "Neither":
            findings.append(
                f"{faster} had lower average end-to-end latency in this run "
                f"(OSS {_fmt_seconds(o_lat)}, Gemini {_fmt_seconds(g_lat)})."
            )

    for label, key in (
        ("Jailbreak resistance", "jailbreak_resistance_rate"),
        ("Memory consistency", "memory_consistency_rate"),
        ("Tool-use success", "tool_use_success_rate"),
    ):
        o_val = oss.get(key)
        g_val = gemini.get(key)
        if o_val is not None and g_val is not None and o_val != g_val:
            better = "OSS" if o_val > g_val else "Gemini"
            findings.append(f"{label}: {better} higher ({_fmt_rate(o_val)} vs {_fmt_rate(g_val)}).")

    g_cost = gemini.get("estimated_cost_per_100_prompts")
    if g_cost is not None:
        findings.append(
            f"Gemini estimated API cost for 100 prompts in this run: {_fmt_cost(g_cost)}; OSS API cost: $0.00."
        )

    if rows:
        gemini_errors = sum(1 for r in rows if r.get("assistant_type") == "gemini" and r.get("error"))
        oss_errors = sum(1 for r in rows if r.get("assistant_type") == "oss" and r.get("error"))
        if gemini_errors or oss_errors:
            findings.append(f"Error rows: OSS {oss_errors}, Gemini {gemini_errors}.")

    if not findings:
        findings.append(
            "Insufficient category coverage in this sample for cross-metric conclusions; run the full 40-prompt eval."
        )

    return findings


if __name__ == "__main__":
    generate_report()
    print(f"Wrote: {REPORT_PATH}")
