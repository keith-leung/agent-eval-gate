"""LangSmith tracing integration — env-var driven, off when key unset."""

from __future__ import annotations

import os
from typing import Optional


def setup_langsmith(api_key: Optional[str] = None) -> None:
    """Enable LangSmith tracing if API key is set."""
    key = api_key or os.environ.get("LANGSMITH_API_KEY", "")
    if not key:
        return
    try:
        os.environ["LANGSMITH_API_KEY"] = key
        # LangSmith's OpenAI SDK integration is env-var driven; no further code needed.
    except Exception:
        pass  # Silently skip if LangSmith SDK not installed


def is_langsmith_enabled() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY", ""))
