"""
Preprocessing and synthetic shot generation for the xG model.

The raw dataset is at player‑season level (aggregated statistics). To train an
expected‑goals model, we require shot‑level events. In a production system this
would come from event‑level tracking data (e.g. Opta). Here we *simulate*
plausible shot events from the season aggregates while documenting all
assumptions explicitly.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from . import config


def _estimate_shots_for_row(row: pd.Series) -> int:
    """
    Heuristic to estimate the number of shots a player took in a season.

    We approximate shots per appearance by position:
    - Forwards take many shots per game.
    - Midfielders take some.
    - Defenders take few.
    - Goalkeepers almost never shoot.

    The exact values are encoded in `config` and can be tuned.
    """
    position = (row.get("player_position") or "").lower()
    apps = max(int(row.get("appearances", 0)), 0)

    if "forward" in position:
        rate = config.SHOTS_PER_APPEARANCE_FORWARD
    elif "midfielder" in position:
        rate = config.SHOTS_PER_APPEARANCE_MIDFIELDER
    elif "defender" in position:
        rate = config.SHOTS_PER_APPEARANCE_DEFENDER
    elif "goalkeeper" in position:
        rate = config.SHOTS_PER_APPEARANCE_GOALKEEPER
    else:
        # Unknown position: fall back to a conservative midfielder‑like rate.
        rate = config.SHOTS_PER_APPEARANCE_MIDFIELDER

    est_shots = int(round(apps * rate))
    return max(est_shots, config.MIN_SHOTS_PER_PLAYER_SEASON)


def _simulate_shot_outcomes(num_shots: int, goals: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate which of a player's shots became goals.

    We assume that each shot has an equal, independent probability of being a goal
    within a season, and that the season conversion rate is:

        p_goal = goals / num_shots.

    This is of course a simplification, but given we only have season‑level
    aggregates it is a reasonable way to connect goals back to individual shots.
    """
    num_shots = int(num_shots)
    goals = max(int(goals), 0)
    if num_shots <= 0:
        return np.zeros(0, dtype=int)

    p_goal = min(goals / num_shots if num_shots > 0 else 0.0, 0.95)
    return rng.binomial(1, p_goal, size=num_shots)


def generate_synthetic_shots(player_season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand player‑season data into a synthetic shot‑level dataset.

    For each player‑season row we:
    1. Estimate the total number of shots using `_estimate_shots_for_row`.
    2. Simulate which shots were goals to match the aggregate goal tally.
    3. For each shot, sample contextual features such as:
       - match minute
       - score difference (from shooter's perspective)
       - whether the shot was under defensive pressure
       - body part used (head vs foot)
       - generic shot type (open play vs set piece)
       - shot location (x, y) on the pitch in metres

    Returns
    -------
    shots_df : pd.DataFrame
        One row per synthetic shot with columns required for feature engineering.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    records = []

    for _, row in player_season_df.iterrows():
        n_shots = _estimate_shots_for_row(row)
        goals = int(row.get("goals", 0))

        outcomes = _simulate_shot_outcomes(n_shots, goals, rng)

        # Player‑level contextual variables used repeatedly.
        base = {
            "team_id": row.get("team_id"),
            "team_name": row.get("team_name"),
            "player_id": row.get("player_id"),
            "player_name": row.get("player_name"),
            "player_position": row.get("player_position"),
            "season": row.get("season"),
        }

        for is_goal in outcomes:
            # --- Temporal context ------------------------------------------------------
            # Sample a match minute: more shots tend to cluster in later phases,
            # but we keep it simple with a slightly skewed distribution.
            minute = int(rng.integers(1, 91))

            # Score difference from the perspective of the shooting team.
            # We bias towards close scorelines.
            score_diff = rng.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])

            # --- Defensive pressure ----------------------------------------------------
            under_pressure = int(rng.random() < config.MEAN_PRESSURE_PROB)

            # --- Body part and shot type ----------------------------------------------
            body_part = rng.choice(["foot", "head"], p=[1 - config.MEAN_HEAD_SHOT_PROB, config.MEAN_HEAD_SHOT_PROB])
            shot_type = rng.choice(["open_play", "set_piece"], p=[0.8, 0.2])

            # --- Shot location ---------------------------------------------------------
            # We generate (x, y) shot locations based on player position:
            # - Forwards shoot closer to goal and more centrally.
            # - Midfielders shoot from a mix of inside/outside the box.
            # - Defenders tend to shoot from longer distances (set pieces).
            pos = (row.get("player_position") or "").lower()
            if "forward" in pos:
                # Cluster inside the box, 8‑20m from goal.
                x = rng.normal(loc=config.PITCH_LENGTH - 15, scale=5)
                y = rng.normal(loc=config.PITCH_WIDTH / 2, scale=10)
            elif "midfielder" in pos:
                x = rng.normal(loc=config.PITCH_LENGTH - 22, scale=8)
                y = rng.normal(loc=config.PITCH_WIDTH / 2, scale=14)
            elif "defender" in pos:
                x = rng.normal(loc=config.PITCH_LENGTH - 28, scale=10)
                y = rng.normal(loc=config.PITCH_WIDTH / 2, scale=18)
            else:  # goalkeepers / unknown
                x = rng.normal(loc=config.PITCH_LENGTH - 30, scale=12)
                y = rng.normal(loc=config.PITCH_WIDTH / 2, scale=20)

            # Clamp positions to within the pitch.
            x = float(np.clip(x, 0, config.PITCH_LENGTH))
            y = float(np.clip(y, 0, config.PITCH_WIDTH))

            rec = {
                **base,
                "match_minute": minute,
                "score_diff": score_diff,
                "under_pressure": under_pressure,
                "body_part": body_part,
                "shot_type": shot_type,
                "x": x,
                "y": y,
                # Target label for the xG model: 1 if the shot was scored.
                "is_goal": int(is_goal),
            }
            records.append(rec)

    shots_df = pd.DataFrame.from_records(records)
    return shots_df


def train_test_split_shots(
    shots_df: pd.DataFrame, test_size: float = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the synthetic shots dataset into train and test partitions.

    We do a stratified split on the `is_goal` label to ensure both
    train and test sets contain a reasonable mix of goals and non‑goals.
    """
    from sklearn.model_selection import train_test_split

    if test_size is None:
        test_size = config.TEST_SIZE

    train_df, test_df = train_test_split(
        shots_df,
        test_size=test_size,
        random_state=config.RANDOM_SEED,
        stratify=shots_df["is_goal"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

