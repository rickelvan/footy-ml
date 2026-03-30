"""
Match-level prediction utilities for the FootyStats dashboard.

This module trains a simple 1X2 (Home/Draw/Away) model from historical,
completed match results and returns human-readable predictions for display
in the interactive dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class MatchModelBundle:
    pipeline: Pipeline
    label_order: List[str]


def load_completed_matches_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("completed")].copy()
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def _outcome_label(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _prep_match_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_team_name"] = out.get("home_team_name", "").astype(str)
    out["away_team_name"] = out.get("away_team_name", "").astype(str)
    out["season_code"] = out.get("season_code", "").astype(str)

    md = out.get("match_date")
    if md is not None and pd.api.types.is_datetime64_any_dtype(md):
        out["match_month"] = md.dt.month.fillna(0).astype(int)
        out["match_dow"] = md.dt.dayofweek.fillna(0).astype(int)
    else:
        out["match_month"] = 0
        out["match_dow"] = 0

    hs = pd.to_numeric(out.get("home_score", 0), errors="coerce").fillna(0)
    aw = pd.to_numeric(out.get("away_score", 0), errors="coerce").fillna(0)
    out["outcome"] = [
        _outcome_label(float(h), float(a)) for h, a in zip(hs.tolist(), aw.tolist())
    ]
    return out


def train_match_outcome_model(df: pd.DataFrame) -> Optional[MatchModelBundle]:
    if df is None or df.empty:
        return None

    d = _prep_match_frame(df)
    if len(d) < 20:
        return None

    X = d[["home_team_name", "away_team_name", "season_code", "match_month", "match_dow"]]
    y = d["outcome"].astype(str)

    cat = ["home_team_name", "away_team_name", "season_code"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="passthrough",
    )

    # Multinomial logistic regression is a strong, simple baseline for 1X2.
    clf = LogisticRegression(
        max_iter=400,
        multi_class="multinomial",
        solver="lbfgs",
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    label_order = list(pipe.named_steps["clf"].classes_)
    return MatchModelBundle(pipeline=pipe, label_order=label_order)


def _time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    if "match_date" in d.columns:
        d = d.sort_values("match_date")
    n = len(d)
    cut = max(1, int(round(n * (1 - test_frac))))
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def build_match_predictions_for_dashboard(
    matches_df: pd.DataFrame,
    *,
    max_rows: int = 18,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Returns:
      - list of match prediction rows (for the dashboard table)
      - small metrics dict for quick display
    """
    d = _prep_match_frame(matches_df)
    train_df, test_df = _time_split(d, test_frac=0.25)
    bundle = train_match_outcome_model(train_df)
    if bundle is None or test_df.empty:
        return [], {"accuracy": 0.0, "n_train": float(len(train_df)), "n_test": float(len(test_df))}

    X_test = test_df[["home_team_name", "away_team_name", "season_code", "match_month", "match_dow"]]
    proba = bundle.pipeline.predict_proba(X_test)
    classes = bundle.label_order

    idx = {c: i for i, c in enumerate(classes)}
    pH = proba[:, idx.get("H", 0)]
    pD = proba[:, idx.get("D", 0)]
    pA = proba[:, idx.get("A", 0)]
    pred = bundle.pipeline.predict(X_test).astype(str)
    actual = test_df["outcome"].astype(str).values

    acc = float((pred == actual).mean()) if len(actual) else 0.0

    tail = test_df.tail(max_rows).copy()
    start = max(0, len(test_df) - len(tail))

    rows: List[Dict] = []
    for j, (_, r) in enumerate(tail.iterrows()):
        i = start + j
        md = r.get("match_date")
        md_str = ""
        if isinstance(md, (datetime, pd.Timestamp)) and pd.notna(md):
            md_str = str(pd.Timestamp(md).strftime("%Y-%m-%d %H:%M"))
        rows.append(
            {
                "match_date": md_str,
                "season_code": str(r.get("season_code", "")),
                "home_team": str(r.get("home_team_name", "")),
                "away_team": str(r.get("away_team_name", "")),
                "home_score": int(r.get("home_score", 0)),
                "away_score": int(r.get("away_score", 0)),
                "p_home": float(pH[i]),
                "p_draw": float(pD[i]),
                "p_away": float(pA[i]),
                "pred": str(pred[i]),
                "actual": str(actual[i]),
                "correct": bool(pred[i] == actual[i]),
            }
        )

    metrics = {"accuracy": acc, "n_train": float(len(train_df)), "n_test": float(len(test_df))}
    return rows, metrics

