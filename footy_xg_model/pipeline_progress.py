"""
Thread-safe progress state for the interactive dashboard server.

Used only when `run_pipeline(track_progress=True)` runs in a background thread
so the browser can poll `/api/status` while training.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "status": "idle",
    "phase": "",
    "message": "",
    "pct": 0,
    "error": None,
}


def reset() -> None:
    with _lock:
        _state.update(
            {
                "status": "idle",
                "phase": "",
                "message": "",
                "pct": 0,
                "error": None,
            }
        )


def start_run() -> None:
    with _lock:
        _state.update(
            {
                "status": "running",
                "phase": "starting",
                "message": "Starting pipeline…",
                "pct": 0,
                "error": None,
            }
        )


def set_progress(phase: str, message: str, pct: int) -> None:
    with _lock:
        if _state["status"] not in ("error", "done"):
            _state["status"] = "running"
        _state["phase"] = phase
        _state["message"] = message
        _state["pct"] = max(0, min(100, int(pct)))


def set_done() -> None:
    with _lock:
        _state["status"] = "done"
        _state["phase"] = "complete"
        _state["message"] = "Training finished — refreshing dashboard…"
        _state["pct"] = 100


def set_error(message: str) -> None:
    with _lock:
        _state["status"] = "error"
        _state["error"] = message
        _state["message"] = message


def snapshot() -> Dict[str, Any]:
    with _lock:
        return dict(_state)
