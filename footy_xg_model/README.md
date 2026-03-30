## FootyStats Unified Analytics Platform

This package combines **season-level player/team analytics** with an
**Expected Goals (xG) shot-probability model** into a single modular pipeline.
One command produces a comprehensive HTML report covering both domains.

### Module Overview

| Module | Responsibility |
|---|---|
| `config.py` | Central configuration (paths, seeds, pitch geometry, feature lists, season analytics thresholds). |
| `data_loader.py` | Load and validate the player-season CSV. |
| `season_analytics.py` | Player performance scores, team aggregations, goal-prediction regression model (Ridge / RF / GB with LOO-CV), next-season projections, discipline risk flags, breakout candidates. |
| `preprocessing.py` | Expand player-season rows into synthetic shot-level events and perform train/test splits. |
| `feature_engineering.py` | Compute shot geometry features (distance, angle) and build feature/target matrices. |
| `train_model.py` | Define xG ML pipelines (Logistic Regression, Random Forest, Gradient Boosting), perform cross-validated hyper-parameter tuning, select and calibrate the best model, save it. |
| `evaluate_model.py` | Compute probability metrics (log loss, ROC-AUC, Brier score, precision/recall), plot diagnostic curves and feature importances. |
| `predict.py` | Load the saved xG pipeline and generate predictions for new shots. |
| `report_generator.py` | Build a self-contained HTML report with all season analytics and xG model results. |
| `main.py` | Orchestrate both pipelines end-to-end. |

### Running the System

From the `footy ml` project root:

```bash
pip install -r requirements.txt
python -u -m footy_xg_model.main
```

> Use the `-u` flag for unbuffered output so progress prints appear in real time.

### What the Pipeline Produces

1. **Season Analytics** -- player feature engineering, team summaries, top 10 performers, goal-prediction model comparison, next-season player and team projections, discipline risk flags, breakout candidates.
2. **xG Shot Model** -- synthetic shot generation, xG model training and tuning, probability calibration, evaluation metrics, diagnostic plots (ROC, PR, confusion matrix, calibration curve), sample xG predictions.
3. **Unified HTML Report** at `footy_xg_model/artifacts/xg_evaluation_report.html` containing everything above in a single self-contained file.
4. **Saved xG pipeline** at `footy_xg_model/models/best_calibrated_xg_pipeline.joblib` for downstream prediction.

### Notes on Data Modelling

Because the provided CSV is at **player-season** level, the system synthesises
per-shot events using transparent, configurable assumptions defined in
`config.py` (e.g. expected shots per appearance by position). All such choices
are heavily commented in the code so they can be reviewed and refined.
