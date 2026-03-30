"""
Data loading utilities for the FootyStats xG system.

Responsibilities:
- Read the raw player‑season CSV file.
- Apply light validation and normalisation.
- Expose a clean pandas DataFrame to the rest of the pipeline.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from . import config


def load_player_season_data(csv_path: Union[str, Path] = None) -> pd.DataFrame:
    """
    Load the player‑season level dataset.

    Parameters
    ----------
    csv_path:
        Optional override for the dataset path. If omitted, the default path
        from `config.RAW_PLAYER_SEASON_PATH` is used.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame containing one row per player‑season with at least:
        - team_id, team_name, player_id, player_name, player_position
        - season, appearances, goals, assists, yellow_cards, red_cards, clean_sheets

    Notes
    -----
    We keep this function intentionally simple. In a production system, we would
    add:
    - schema validation (e.g. with pandera or pydantic)
    - stronger type coercion and missing‑value reporting
    - logging instrumentation
    """
    # Resolve the path using the configuration module so that
    # the entire project respects a single source of truth.
    path = Path(csv_path) if csv_path is not None else config.RAW_PLAYER_SEASON_PATH

    if not path.exists() and csv_path is None:
        name = path.name
        for candidate in (
            config.PROJECT_ROOT / name,
            config.PROJECT_ROOT / "1" / name,
        ):
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(
            f"Player‑season dataset not found at {path}. "
            f"Place {path.name!r} in {config.PROJECT_ROOT} or {config.PROJECT_ROOT / '1'}."
        )

    df = pd.read_csv(path)

    # Basic sanity checks and coercions.
    expected_columns = {
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "player_position",
        "season",
        "appearances",
        "goals",
    }
    missing = expected_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset at {path} is missing required columns: {missing}")

    # Ensure numeric columns are of numeric dtype; errors coerced to NaN and later filled.
    numeric_cols = [
        "team_id",
        "player_id",
        "season",
        "appearances",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
        "clean_sheets",
        "saves",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Light cleaning: drop completely empty rows and fill obvious missing values.
    df = df.dropna(how="all")
    df["appearances"] = df["appearances"].fillna(0).astype(int)
    df["goals"] = df["goals"].fillna(0).astype(int)

    # Normalise position labels for easier downstream use.
    if "player_position" in df.columns:
        df["player_position"] = df["player_position"].str.strip().str.title()

    return df

