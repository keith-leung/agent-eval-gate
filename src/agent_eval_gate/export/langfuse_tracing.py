"""Langfuse tracing integration."""

from __future__ import annotations

import os
from typing import Optional


def setup_langfuse(public_key: Optional[str] = None, secret_key: Optional[str] = None) -> None:
    pk = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not pk or not sk:
        return
    try:
        os.environ["LANGFUSE_PUBLIC_KEY"] = pk
        os.environ["LANGFUSE_SECRET_KEY"] = sk
    except Exception:
        pass


def is_langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "")) and bool(os.environ.get("LANGFUSE_SECRET_KEY", ""))
