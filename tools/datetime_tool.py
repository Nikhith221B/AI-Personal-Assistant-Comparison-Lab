"""Current local date/time tool."""

from __future__ import annotations

from datetime import datetime


def current_datetime_iso() -> str:
    """Return current local date and time as an ISO string."""
    return datetime.now().isoformat(timespec="seconds")
