"""
Prediction utilities for the FootyStats xG system.

This module exposes a simple, documented interface for generating xG values
for new shots using a trained and calibrated pipeline.
"""

from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

from . import config
from .feature_engineering import add_geometric_features, build_feature_target_matrices


def load_trained_pipeline(path: Path = None):
    """
    Load the previously saved xG pipeline from disk.

    The pipeline encapsulates both preprocessing (feature scaling / encoding)
    and the trained probabilistic classifier.
    """
    if path is None:
        path = config.BEST_MODEL_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained xG pipeline not found at {path}. Have you run main.py to train it?"
        )
    return joblib.load(path)


def predict_xg_for_shots(pipeline, shots_df: pd.DataFrame) -> np.ndarray:
    """
    Compute xG for a batch of shot events represented as a DataFrame.

    The function:
    - adds geometric features (distance, angle)
    - ensures the feature set matches the training pipeline
    - returns an array of probabilities in [0, 1]
    """
    enriched = add_geometric_features(shots_df)
    X, _ = build_feature_target_matrices(enriched.assign(is_goal=0))
    proba = pipeline.predict_proba(X)[:, 1]
    return proba


def pretty_print_predictions(shots_df: pd.DataFrame, xg: Iterable[float]) -> None:
    """
    Print human‑readable xG predictions for a set of shots.
    """
    for idx, (shot, p) in enumerate(zip(shots_df.to_dict(orient="records"), xg), start=1):
        desc = (
            f"Shot {idx} | Player: {shot.get('player_name', 'N/A')} | "
            f"Position: ({shot.get('x'):.1f}m, {shot.get('y'):.1f}m) | "
            f"Body part: {shot.get('body_part')} | "
            f"Minute: {shot.get('match_minute')}"
        )
        print(f"{desc}  ->  xG = {p:.2f}")

