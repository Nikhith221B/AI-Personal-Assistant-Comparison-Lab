# Evaluation Report

*Generated: 2026-05-25 09:50 UTC*

For setup, architecture, deployment, and tradeoffs, see [README.md](../README.md).

## Executive summary

This lab compared **local OSS** (`Qwen/Qwen2.5-0.5B-Instruct`) against **frontier API** (`gemini-2.5-flash`) on the same prompts through the shared pipeline (guardrails, tools, memory).

> **Data source:** reports/results.json (2026-05-25T09:50:04.974120+00:00)

**Sample:** 40 prompts × 2 assistants (80 rows); categories: bias, factual, jailbreak, memory, tool_use.

**Headline (directional):**
- Factual pass rate: OSS 90.0% vs Gemini 100.0%.
- End-to-end latency: OSS 8.470s vs Gemini 1.804s.
- Est. cost / 100 prompts: OSS $0.0000 vs Gemini $0.0058.

**Disclaimer:** Directional demo metrics only (small per-category sample sizes); not statistically significant.

## Model comparison

| Metric | OSS | Gemini |
|--------|-----|--------|
| Factual pass rate | 90.0% | 100.0% |
| Jailbreak resistance | 90.0% | 90.0% |
| Bias safety pass | 100.0% | 100.0% |
| Content safety (combined) | 95.0% | 95.0% |
| Memory consistency | 100.0% | 100.0% |
| Tool-use success | 100.0% | 100.0% |
| Rows scored | 40 | 40 |

## Metric bars (higher bar = better on this chart)

| Dimension | OSS | Gemini |
|-----------|-----|--------|
| Factual accuracy | █████████░ 90% | ██████████ 100% |
| Jailbreak resistance | █████████░ 90% | █████████░ 90% |
| Bias safety | ██████████ 100% | ██████████ 100% |
| Memory consistency | ██████████ 100% | ██████████ 100% |
| Tool use | ██████████ 100% | ██████████ 100% |

## Cost and latency

| Assistant | End-to-end latency | Model-only latency | Est. cost / 100 prompts |
|-----------|-------------------|--------------------|-------------------------|
| OSS | 8.470s | 11.293s | $0.0000 |
| Gemini | 1.804s | 2.405s | $0.0058 |

*OSS cost is $0 API; excludes local hardware/electricity. Gemini cost uses token estimates in `evals/metrics.py`.*

## Run health

- **OSS errors:** No errors recorded.
- **Gemini errors:** No errors recorded.

## OSS factual failures (sample)

OSS factual rows use Gemini as an optional LLM judge when enabled; if the judge returns no verdict or errors, scoring falls back to the same fuzzy_match rules as Gemini (expected answer need not match exact wording).

- `fact_09`: LLM judge verdict: FAIL — response: "Plants primarily absorb carbon dioxide (CO2) from the atmosphere for photosynthesis. This process is called cellular ..."

## Key findings

- On factual prompts in this sample, Gemini passed more often (100.0% vs 90.0%).
- Gemini had lower average end-to-end latency in this run (OSS 8.470s, Gemini 1.804s).
- Gemini estimated API cost for 100 prompts in this run: $0.0058; OSS API cost: $0.00.

## Recommendation

- **Prefer OSS (Qwen 0.5B)** for zero marginal API cost, offline/local demos, and Hugging Face Spaces where only the small model is deployed. Expect weaker reasoning, safety, and memory on hard prompts.
- **Prefer Gemini** when answer quality, safety refusals, and memory/tool reliability matter more than cost and latency — at the price of API quotas and per-token spend.
- **Use this repo’s pipeline for both** so comparisons reflect guardrails and tools, not raw model-only behavior.
