"""Hugging Face Spaces (OSS-only) deployment checks."""

from __future__ import annotations

import importlib
import os
import sys


def _reload_app(monkeypatch, *, deployment_mode: str, gemini_key: str | None = None) -> object:
    if gemini_key is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
    monkeypatch.setenv("DEPLOYMENT_MODE", deployment_mode)
    if "app" in sys.modules:
        return importlib.reload(sys.modules["app"])
    import app

    return app


def test_spaces_mode_builds_without_gemini_key(monkeypatch) -> None:
    app = _reload_app(monkeypatch, deployment_mode="spaces")
    assert app.IS_SPACES_MODE is True
    demo = app.build_ui()
    assert demo is not None


def test_local_mode_builds_without_gemini_key(monkeypatch) -> None:
    app = _reload_app(monkeypatch, deployment_mode="local")
    assert app.IS_SPACES_MODE is False
    demo = app.build_ui()
    assert demo is not None
