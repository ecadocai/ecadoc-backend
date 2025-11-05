"""
Lightweight progress bus for emitting non-sensitive, fine-grained events from
agent tools during execution. The streaming API attaches a callback for the
current request; tool functions can call `emit` to publish status updates.

This uses a contextvar so concurrent requests do not leak callbacks across
requests. Tools should only emit small, textual payloads (no PII).
"""
from __future__ import annotations

from typing import Callable, Optional, Dict, Any
from contextvars import ContextVar

_progress_cb: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar(
    "progress_cb", default=None
)

def set_progress_callback(cb: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Attach a progress callback for the current task context."""
    _progress_cb.set(cb)

def emit(code: str, **data: Any) -> None:
    """Emit a progress event if a callback is present.

    Parameters
    - code: short machine code for the event, e.g. 'page_image_start'
    - **data: additional structured fields (e.g. page=3, count=12)
    """
    cb = _progress_cb.get()
    if cb is None:
        return
    try:
        payload: Dict[str, Any] = {"code": code}
        payload.update(data)
        cb(payload)
    except Exception:
        # Never allow progress reporting to break the main flow
        pass

