"""
Configuration module for the FootyStats xG modelling system.

This centralises all configuration so that:
- experiments are reproducible
- paths and hyperparameters are not hard‑coded across the codebase
- it is easy to tweak assumptions about the football model
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------------------------------------------

# Global random seed used across NumPy / pandas / scikit‑learn.
RANDOM_SEED: int = 42


# --------------------------------------------------------------------------------------
# DATA PATHS
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the original player‑season level dataset (project root or `1/` — see data_loader fallback).
RAW_PLAYER_SEASON_PATH = PROJECT_ROOT / "footy-dataset-20260316-103939.csv"

# Directory where intermediate, engineered datasets will be stored.
DATA_OUT_DIR = PROJECT_ROOT / "footy_xg_model" / "artifacts"

# Directory where trained models and preprocessing pipelines will be saved.
MODEL_OUT_DIR = PROJECT_ROOT / "footy_xg_model" / "models"

# Optional headshots for the discipline watchlist (and similar UIs). Filenames
# should match ``player_name`` after normalization — see ``player_photos.py``.
PLAYERS_PHOTOS_DIR = PROJECT_ROOT / "players"


# --------------------------------------------------------------------------------------
# FOOTBALL / PITCH GEOMETRY
# --------------------------------------------------------------------------------------

# We assume standard FIFA pitch dimensions in metres.
PITCH_LENGTH: float = 105.0  # goal to goal
PITCH_WIDTH: float = 68.0

# We work in an attacking‑direction coordinate system where:
# - (0, PITCH_WIDTH / 2) is the centre of the defending goal
# - (PITCH_LENGTH, PITCH_WIDTH / 2) is the centre of the attacking goal
# All generated shots are taken against the attacking goal at x = PITCH_LENGTH.


# --------------------------------------------------------------------------------------
# SYNTHETIC SHOT GENERATION (FROM PLAYER‑SEASON STATS)
# --------------------------------------------------------------------------------------

# The supplied CSV is at player‑season granularity, not shot level.
# To train an xG model we *simulate* shot‑level events from this summary data.
# These assumptions are deliberately simple but football‑plausible and
# fully documented so they can be refined later.

# Approximate number of shots per appearance by player position.
SHOTS_PER_APPEARANCE_FORWARD: float = 2.5
SHOTS_PER_APPEARANCE_MIDFIELDER: float = 1.0
SHOTS_PER_APPEARANCE_DEFENDER: float = 0.3
SHOTS_PER_APPEARANCE_GOALKEEPER: float = 0.05

# Minimum number of shots we allow per player‑season so that players
# with very few minutes still contribute some training signal.
MIN_SHOTS_PER_PLAYER_SEASON: int = 3

# Distributional assumptions about contextual variables when generating
# synthetic shots. All can be refined based on domain knowledge or data.
MEAN_PRESSURE_PROB: float = 0.35  # probability that a shot is taken under pressure
MEAN_HEAD_SHOT_PROB: float = 0.18  # probability of a header vs foot shot


# --------------------------------------------------------------------------------------
# MODELLING / TRAINING CONFIG
# --------------------------------------------------------------------------------------

TEST_SIZE: float = 0.2  # held‑out test set proportion
CV_FOLDS: int = 5       # k‑fold cross‑validation for hyper‑parameter tuning

# Feature groups used in the modelling pipeline.
NUMERIC_FEATURES = [
    "shot_distance",
    "shot_angle",
    "match_minute",
    "score_diff",
]

CATEGORICAL_FEATURES = [
    "body_part",
    "shot_type",
    "player_position",
]

# Path where the best calibrated model (and its preprocessing pipeline)
# will be saved. We persist the *entire* pipeline using joblib so that
# downstream prediction only needs raw feature columns.
BEST_MODEL_PATH = MODEL_OUT_DIR / "best_calibrated_xg_pipeline.joblib"

# Path for the HTML evaluation report (self-contained, with embedded plots).
REPORT_PATH = DATA_OUT_DIR / "xg_evaluation_report.html"

# Path for the interactive ML visualization dashboard.
DASHBOARD_PATH = DATA_OUT_DIR / "ml_dashboard.html"


# --------------------------------------------------------------------------------------
# SEASON ANALYTICS (Player-season regression model & projections)
# --------------------------------------------------------------------------------------

NEXT_SEASON_GAMES: int = 12

# Weights for the composite player performance score.
PERFORMANCE_WEIGHTS = {
    "goals": 3.0,
    "assists": 1.5,
    "involvement": 1.0,
    "discipline": -0.2,
}

# Breakout candidate thresholds.
BREAKOUT_MIN_GPG: float = 0.3   # minimum goals-per-game
BREAKOUT_MAX_APPS: int = 6      # maximum appearances

# Discipline risk threshold (projected discipline score).
DISCIPLINE_RISK_THRESHOLD: float = 2.0


# --------------------------------------------------------------------------------------
# PLOTTING CONFIG
# --------------------------------------------------------------------------------------

FIGURE_DPI: int = 120

