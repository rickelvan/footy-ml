"""
Season-level analytics for the FootyStats platform.

This module encapsulates the player-season analytics pipeline that was
originally in ``footy_ml_model.py``: per-game rate features, composite
performance scoring, team aggregations, a regression model that predicts
season goals, next-season projections, discipline risk flags, and breakout
candidate detection.

All configurable thresholds and weights live in ``config.py`` so they can be
tuned without touching this code.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder

from . import config


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SeasonAnalyticsResult:
    """Bundle every artefact the season analytics pipeline produces."""

    player_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_performers: pd.DataFrame = field(default_factory=pd.DataFrame)

    best_model_name: str = ""
    best_model_mae: float = 0.0
    model_comparison: List[Dict[str, Any]] = field(default_factory=list)
    goal_feature_importances: Dict[str, float] = field(default_factory=dict)

    player_projections: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_projections: pd.DataFrame = field(default_factory=pd.DataFrame)

    discipline_risks: pd.DataFrame = field(default_factory=pd.DataFrame)
    breakout_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)

    num_players: int = 0
    num_teams: int = 0
    total_goals: int = 0
    total_assists: int = 0


# ---------------------------------------------------------------------------
# Player-level feature engineering
# ---------------------------------------------------------------------------

def compute_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-game rates, discipline score, involvement index, and a
    weighted composite performance score for every player.
    """
    df = df.copy()

    numeric_cols = [
        "appearances", "goals", "assists",
        "yellow_cards", "red_cards", "clean_sheets", "saves",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["appearances"] > 0].copy()

    df["goals_per_game"] = df["goals"] / df["appearances"]
    df["assists_per_game"] = df["assists"] / df["appearances"]
    df["goal_contributions_per_game"] = (df["goals"] + df["assists"]) / df["appearances"]
    df["discipline_score"] = df["yellow_cards"] + df["red_cards"] * 3
    df["involvement"] = df["appearances"] / df["appearances"].max()

    enc = LabelEncoder()
    df["position_enc"] = enc.fit_transform(df["player_position"].fillna("Unknown"))

    w = config.PERFORMANCE_WEIGHTS
    df["performance_score"] = (
        df["goals_per_game"]   * w["goals"]
        + df["assists_per_game"] * w["assists"]
        + df["involvement"]      * w["involvement"]
        + df["discipline_score"] * w["discipline"]
    )

    return df


# ---------------------------------------------------------------------------
# Team aggregations
# ---------------------------------------------------------------------------

def compute_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level data to team totals."""
    team = df.groupby("team_name").agg(
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_apps=("appearances", "sum"),
        avg_discipline=("discipline_score", "mean"),
        squad_size=("player_id", "count"),
    ).reset_index()
    team["team_goals_per_app"] = team["total_goals"] / team["total_apps"]
    return team.sort_values("total_goals", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Goal-prediction regression model
# ---------------------------------------------------------------------------

_FEATURES = [
    "appearances", "assists", "yellow_cards", "red_cards",
    "position_enc", "assists_per_game", "involvement", "discipline_score",
]


def train_goal_prediction_model(
    df: pd.DataFrame,
) -> Tuple[Any, str, float, List[Dict[str, Any]], Dict[str, float]]:
    """
    Train Ridge / Random-Forest / Gradient-Boosting regressors on the
    player-season data to predict total season goals.  Model selection is
    performed via Leave-One-Out cross-validation (best for small datasets).

    Returns
    -------
    best_model : fitted estimator
    best_name  : str
    best_mae   : float
    comparison : list of {"name", "mae"} dicts for every candidate
    feat_imp   : dict mapping feature name -> importance value
    """
    X = df[_FEATURES].values
    y = df["goals"].values

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=4, random_state=config.RANDOM_SEED,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=config.RANDOM_SEED,
        ),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    loo = LeaveOneOut()
    comparison: List[Dict[str, Any]] = []
    best_model, best_name, best_mae = None, "", 9999.0

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=loo, scoring="neg_mean_absolute_error",
        )
        mae = -scores.mean()
        comparison.append({"name": name, "mae": round(mae, 4)})
        if mae < best_mae:
            best_mae, best_name, best_model = mae, name, model

    best_model.fit(X, y)

    feat_imp: Dict[str, float] = {}
    if hasattr(best_model, "feature_importances_"):
        for feat, imp in zip(_FEATURES, best_model.feature_importances_):
            feat_imp[feat] = float(imp)
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_)
        for feat, c in zip(_FEATURES, coefs):
            feat_imp[feat] = float(c)

    return best_model, best_name, best_mae, comparison, feat_imp


# ---------------------------------------------------------------------------
# Next-season projections
# ---------------------------------------------------------------------------

def compute_projections(
    df: pd.DataFrame,
    model: Any,
    team_stats: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project next-season goals, assists, appearances, and goal-contributions
    for every player and aggregate to team level.  Also computes a
    year-over-year trend for team goals.
    """
    nsg = config.NEXT_SEASON_GAMES
    proj = df.copy()

    proj["proj_appearances"] = (
        (proj["involvement"] * nsg).clip(upper=nsg).round(0).astype(int)
    )
    proj["proj_assists"] = (proj["assists_per_game"] * proj["proj_appearances"]).round(1)
    proj["proj_yellow"] = (
        proj["yellow_cards"] / proj["appearances"] * proj["proj_appearances"]
    ).round(1)
    proj["proj_red"] = (
        proj["red_cards"] / proj["appearances"] * proj["proj_appearances"]
    ).round(1)
    proj["proj_discipline"] = proj["proj_yellow"] + proj["proj_red"] * 3
    proj["proj_involvement"] = proj["proj_appearances"] / proj["proj_appearances"].max()
    proj["proj_assists_per_game"] = proj["assists_per_game"]

    X_proj = proj[[
        "proj_appearances", "proj_assists", "proj_yellow", "proj_red",
        "position_enc", "proj_assists_per_game", "proj_involvement", "proj_discipline",
    ]].values

    proj["projected_goals"] = model.predict(X_proj).clip(min=0).round(1)
    proj["projected_goal_contributions"] = proj["projected_goals"] + proj["proj_assists"]

    # Team-level projections with trend arrows.
    team_proj = proj.groupby("team_name").agg(
        proj_goals=("projected_goals", "sum"),
        proj_assists=("proj_assists", "sum"),
        proj_apps=("proj_appearances", "sum"),
    ).reset_index().round(1)
    team_proj["proj_goals_per_game"] = (
        team_proj["proj_goals"] / team_proj["proj_apps"]
    ).round(3)
    team_proj = team_proj.sort_values("proj_goals", ascending=False).reset_index(drop=True)

    goals_current = team_stats.set_index("team_name")["total_goals"].to_dict()
    trends = []
    for _, row in team_proj.iterrows():
        prev = goals_current.get(row["team_name"], 0)
        pct = ((row["proj_goals"] - prev) / prev * 100) if prev > 0 else 0
        trends.append(round(pct, 1))
    team_proj["trend_pct"] = trends

    return proj, team_proj


# ---------------------------------------------------------------------------
# Risk & breakout detection
# ---------------------------------------------------------------------------

def find_discipline_risks(proj_df: pd.DataFrame) -> pd.DataFrame:
    """Return players whose projected discipline score exceeds the threshold."""
    threshold = config.DISCIPLINE_RISK_THRESHOLD
    risk = proj_df[proj_df["proj_discipline"] >= threshold][[
        "player_name", "team_name", "player_position",
        "proj_yellow", "proj_red", "proj_discipline",
    ]].copy()
    return risk.sort_values("proj_discipline", ascending=False).reset_index(drop=True)


def find_breakout_candidates(proj_df: pd.DataFrame) -> pd.DataFrame:
    """Identify players with high goals-per-game but few appearances."""
    bc = proj_df[
        (proj_df["goals_per_game"] >= config.BREAKOUT_MIN_GPG)
        & (proj_df["appearances"] <= config.BREAKOUT_MAX_APPS)
    ][[
        "player_name", "team_name", "player_position",
        "appearances", "goals", "goals_per_game", "projected_goals",
    ]].copy()
    return bc.sort_values("goals_per_game", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_season_analytics(player_season_df: pd.DataFrame) -> SeasonAnalyticsResult:
    """
    Execute the full season-analytics pipeline and return a result bundle
    that ``main.py`` feeds into the unified report generator.
    """
    print("Running season-level analytics...")

    df = compute_player_features(player_season_df)
    team_stats = compute_team_stats(df)
    top_performers = df.nlargest(10, "performance_score")[[
        "player_name", "team_name", "player_position",
        "appearances", "goals", "assists", "performance_score",
    ]].round(3).reset_index(drop=True)

    print("  Training goal-prediction model (LOO-CV)...")
    model, best_name, best_mae, comparison, feat_imp = train_goal_prediction_model(df)

    print("  Computing next-season projections...")
    proj_df, team_proj = compute_projections(df, model, team_stats)

    discipline = find_discipline_risks(proj_df)
    breakouts = find_breakout_candidates(proj_df)

    result = SeasonAnalyticsResult(
        player_df=df,
        team_stats=team_stats,
        top_performers=top_performers,
        best_model_name=best_name,
        best_model_mae=best_mae,
        model_comparison=comparison,
        goal_feature_importances=feat_imp,
        player_projections=proj_df,
        team_projections=team_proj,
        discipline_risks=discipline,
        breakout_candidates=breakouts,
        num_players=len(df),
        num_teams=df["team_name"].nunique(),
        total_goals=int(df["goals"].sum()),
        total_assists=int(df["assists"].sum()),
    )
    print("  Season analytics complete.")
    return result
