"""Safe refusal messages for blocked prompts or model outputs.

This is a lightweight rule-based layer, not a complete safety system.
"""

from __future__ import annotations

GENERIC_REFUSAL = (
    "I can't help with that request. "
    "I'm designed to be helpful, harmless, and honest. "
    "Please ask something else."
)

_CATEGORY_REFUSALS: dict[str, str] = {
    "malware": (
        "I can't help create, spread, or improve malware or hacking tools. "
        "I can explain cybersecurity defense at a high level if that helps."
    ),
    "phishing": (
        "I can't help with phishing or deceptive schemes to steal information. "
        "I can share general tips on recognizing phishing instead."
    ),
    "credential_theft": (
        "I can't help steal credentials or break into accounts. "
        "I can suggest legitimate account security practices."
    ),
    "weapons": (
        "I can't provide instructions for weapons or explosives. "
        "Please ask about a safe, legal topic."
    ),
    "violent_wrongdoing": (
        "I can't help with violence or harming people. "
        "If you are in danger, contact local emergency services."
    ),
    "hate_discrimination": (
        "I won't produce hateful or discriminatory content. "
        "I'm happy to discuss respectful, inclusive topics instead."
    ),
    "self_harm": (
        "I can't help with self-harm. "
        "If you are struggling, please reach out to a trusted person or a local crisis helpline."
    ),
    "illegal_activity": (
        "I can't help with illegal activity. "
        "Please ask about lawful alternatives."
    ),
}


def get_refusal(category: str | None = None) -> str:
    """Return a refusal message for a safety category."""
    if category and category in _CATEGORY_REFUSALS:
        return _CATEGORY_REFUSALS[category]
    return GENERIC_REFUSAL
