"""Open-source Qwen2.5 assistant via Hugging Face Transformers."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from assistants.base import BaseAssistant

OSS_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9

_model = None
_tokenizer = None
_load_error: str | None = None
_loading = False
_load_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def is_model_loaded() -> bool:
    """True if the OSS model and tokenizer are cached."""
    return _model is not None and _tokenizer is not None


def is_model_loading() -> bool:
    """True while the model is being downloaded or loaded."""
    return _loading


def get_load_error() -> str | None:
    """Return the last model load error message, if any."""
    return _load_error


def _generation_timeout_seconds() -> float:
    raw = os.getenv("OSS_GENERATION_TIMEOUT_SECONDS", "120")
    try:
        return float(raw)
    except ValueError:
        return 120.0


def _ensure_loaded() -> None:
    """Lazy-load and cache the model on first use."""
    global _model, _tokenizer, _load_error, _loading

    if _model is not None and _tokenizer is not None:
        return
    if _load_error is not None:
        raise RuntimeError(_load_error)

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return
        if _load_error is not None:
            raise RuntimeError(_load_error)

        _loading = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(OSS_MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                OSS_MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.to("cpu")
            model.eval()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _tokenizer = tokenizer
            _model = model
        except Exception as exc:
            _load_error = f"Failed to load {OSS_MODEL_ID}: {exc}"
            raise RuntimeError(_load_error) from exc
        finally:
            _loading = False


def _generate_text(messages: list[dict[str, str]]) -> str:
    """Run inference (must be called with model already loaded)."""
    import torch

    assert _model is not None and _tokenizer is not None

    prompt_text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            do_sample=True,
            pad_token_id=_tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text or "(No response generated.)"


class OSSAssistant(BaseAssistant):
    assistant_type = "oss"

    @property
    def model_name(self) -> str:
        return OSS_MODEL_ID

    def _generate_model_response(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del kwargs
        _ensure_loaded()

        timeout = _generation_timeout_seconds()
        future = _executor.submit(_generate_text, messages)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"OSS generation exceeded {timeout:.0f}s. "
                "Try a shorter message or increase OSS_GENERATION_TIMEOUT_SECONDS."
            ) from exc
