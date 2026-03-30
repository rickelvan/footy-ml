"""
Feature engineering for the FootyStats xG model.

This module transforms raw shot‑level data into features that are both
football‑meaningful and amenable to machine learning algorithms.

Key ideas:
- Represent shot geometry via distance and angle to the centre of the goal.
- Include contextual variables such as match minute and score difference.
- Keep the interface model‑agnostic so multiple algorithms can reuse the
  same engineered feature set.
"""

import math
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


def compute_shot_distance_and_angle(
    x: float, y: float, goal_x: float, goal_y: float
) -> Tuple[float, float]:
    """
    Compute distance and angle from the shot location to the centre of the goal.

    Geometry:
    ---------
    - The distance is the straight‑line Euclidean distance between (x, y) and
      the goal centre (goal_x, goal_y).
    - The angle is the opening angle between the lines from the shot point to
      the two goalposts. For simplicity we approximate the goal as a point and
      instead compute the *bearing* angle between the shot and goal centre:

          angle = arctan2(|y - goal_y|, goal_x - x)

      This yields an angle in radians between 0 and pi/2. Larger angles
      correspond to more central shots (better scoring chances).

    Returns
    -------
    distance_m : float
        Distance in metres.
    angle_rad : float
        Angle in radians.
    """
    dx = goal_x - x
    dy = y - goal_y

    distance = math.sqrt(dx * dx + dy * dy)

    # Absolute vertical offset from goal centre; atan2 ensures numerical
    # stability even when dx is very small.
    angle = math.atan2(abs(dy), max(dx, 1e-6))

    return float(distance), float(angle)


def add_geometric_features(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add shot‑geometry features (distance and angle) to the shots DataFrame.

    We assume the attacking goal is located at (PITCH_LENGTH, PITCH_WIDTH / 2).
    """
    goal_x = config.PITCH_LENGTH
    goal_y = config.PITCH_WIDTH / 2.0

    distances = []
    angles = []
    for x, y in zip(shots_df["x"].values, shots_df["y"].values):
        d, a = compute_shot_distance_and_angle(x, y, goal_x, goal_y)
        distances.append(d)
        angles.append(a)

    shots_df = shots_df.copy()
    shots_df["shot_distance"] = distances
    shots_df["shot_angle"] = angles
    return shots_df


def build_feature_target_matrices(
    shots_df: pd.DataFrame,
):
    """
    Construct the feature matrix X and target vector y for modelling.

    This function defines *which* columns are considered features and
    which column is the binary target.

    Returns
    -------
    X : pd.DataFrame
        Feature columns only.
    y : pd.Series
        Binary target (1 = goal, 0 = miss).
    """
    feature_cols = (
        config.NUMERIC_FEATURES
        + config.CATEGORICAL_FEATURES
        + ["under_pressure"]
    )
    missing = [c for c in feature_cols if c not in shots_df.columns]
    if missing:
        raise ValueError(f"Shots DataFrame is missing required feature columns: {missing}")

    X = shots_df[feature_cols].copy()

    # `under_pressure` is a binary numeric feature and is therefore treated
    # as numeric by the modelling pipeline even though it is conceptually
    # categorical. Keeping it numeric avoids an unnecessary one‑hot column.
    y = shots_df["is_goal"].astype(int)
    return X, y

