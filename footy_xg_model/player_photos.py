"""
Match player names to image files in ``config.PLAYERS_PHOTOS_DIR`` and copy
matches into ``artifacts/players/`` so the dashboard can reference them as
``players/<filename>``. Used for the discipline watchlist and breakout candidates.

Naming: the file stem (name without extension) should match the dashboard
``player_name`` after normalization — lowercase, non-alphanumeric collapsed to
a single underscore. Examples:

- Player ``John Smith`` → ``john_smith.jpg``, ``John Smith.png``, etc.
- Player ``Mbappé`` → normalize to ``mbapp`` (accent stripped) — prefer ASCII
  filenames like ``mbappe.jpg`` if needed.

Supported extensions: .jpg, .jpeg, .png, .webp, .gif
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def normalize_player_key(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name.strip())
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "player"


def _photo_index(source_dir: Path) -> Dict[str, Path]:
    """First matching file wins for each normalized stem (sorted for stability)."""
    idx: Dict[str, Path] = {}
    if not source_dir.is_dir():
        return idx
    for p in sorted(source_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTS:
            continue
        key = normalize_player_key(p.stem)
        if key and key not in idx:
            idx[key] = p
    return idx


def attach_photos_to_rows(
    rows: List[Dict[str, Any]],
    *,
    name_field: str = "player_name",
) -> List[Dict[str, Any]]:
    """
    Copy matched images into ``artifacts/players`` and set ``photo`` on each row
    to a relative URL path (``players/...``). ``name_field`` is the dict key for
    the display name used to match filenames.
    """
    source = Path(config.PLAYERS_PHOTOS_DIR)
    dest_root = Path(config.DATA_OUT_DIR) / "players"
    dest_root.mkdir(parents=True, exist_ok=True)
    index = _photo_index(source)

    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        key = normalize_player_key(str(r.get(name_field) or ""))
        rel: Optional[str] = None
        if key and key in index:
            src = index[key]
            ext = src.suffix.lower()
            dest_name = f"{key}{ext}"
            dest_path = dest_root / dest_name
            try:
                shutil.copy2(src, dest_path)
                rel = f"players/{dest_name}"
            except OSError:
                rel = None
        r["photo"] = rel
        out.append(r)
    return out


def attach_photos_to_discipline_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backward-compatible alias for discipline watchlist rows."""
    return attach_photos_to_rows(rows, name_field="player_name")
