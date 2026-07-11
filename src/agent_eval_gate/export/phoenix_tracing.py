"""Phoenix / Arize Phoenix tracing integration."""

from __future__ import annotations

import os
from typing import Optional


def setup_phoenix(endpoint: Optional[str] = None) -> None:
    ep = endpoint or os.environ.get("PHOENIX_ENDPOINT", "")
    if not ep:
        return
    try:
        os.environ["PHOENIX_ENDPOINT"] = ep
    except Exception:
        pass


def is_phoenix_enabled() -> bool:
    return bool(os.environ.get("PHOENIX_ENDPOINT", ""))
