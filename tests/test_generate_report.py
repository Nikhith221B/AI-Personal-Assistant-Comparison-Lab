"""Tests for evaluation report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.generate_report import generate_report


@pytest.fixture
def sample_results(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()

    rows = [
        {
            "prompt_id": "fact_01",
            "category": "factual",
            "assistant_type": "oss",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "Capital of France?",
            "response": "Paris",
            "latency_seconds": 2.0,
            "latency_model_seconds": 2.0,
            "safety_triggered": False,
            "tool_used": "",
            "error": "",
            "score": 1.0,
            "pass_fail": "pass",
            "notes": "",
            "scoring_method": "rule",
        },
        {
            "prompt_id": "fact_01",
            "category": "factual",
            "assistant_type": "gemini",
            "model_name": "gemini-2.5-flash",
            "prompt": "Capital of France?",
            "response": "Paris",
            "latency_seconds": 0.5,
            "latency_model_seconds": 0.5,
            "safety_triggered": False,
            "tool_used": "",
            "error": "",
            "score": 1.0,
            "pass_fail": "pass",
            "notes": "",
            "scoring_method": "rule",
        },
    ]
    payload = {"generated_at": "2026-05-24T00:00:00+00:00", "results": rows}
    (reports / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    import evals.generate_report as mod

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_JSON", reports / "results.json")
    monkeypatch.setattr(mod, "RESULTS_CSV", reports / "results.csv")
    monkeypatch.setattr(mod, "EVAL_LOG_PATH", logs / "eval_runs.jsonl")
    monkeypatch.setattr(mod, "REPORT_PATH", reports / "evaluation_report.md")
    monkeypatch.setattr(mod, "CHARTS_DIR", reports / "charts")

    return reports


def test_generate_report_from_results(sample_results):
    text = generate_report(write_charts=False)
    assert "Executive summary" in text
    assert "Qwen" in text
    assert "gemini" in text.lower()
    assert "Directional demo metrics" in text
    assert (sample_results / "evaluation_report.md").exists()


def test_pending_when_no_data(tmp_path, monkeypatch):
    import evals.generate_report as mod

    reports = tmp_path / "reports"
    reports.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_JSON", reports / "results.json")
    monkeypatch.setattr(mod, "RESULTS_CSV", reports / "results.csv")
    monkeypatch.setattr(mod, "EVAL_LOG_PATH", logs / "eval_runs.jsonl")
    monkeypatch.setattr(mod, "REPORT_PATH", reports / "evaluation_report.md")

    text = generate_report(write_charts=False)
    assert "Pending evaluation run" in text
