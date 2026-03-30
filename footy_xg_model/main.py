"""
Entry point for the FootyStats unified analytics pipeline.

Running this module will:
 1. Load the raw player-season data.
 2. Run season-level analytics (performance scores, goal-prediction model,
    projections, risk flags, breakout candidates).
 3. Generate synthetic shot-level events.
 4. Engineer football-aware features (geometry, context).
 5. Train multiple probabilistic xG models and perform hyper-parameter tuning.
 6. Select and calibrate the best xG model.
 7. Evaluate the final xG model and save artefacts.
 8. Produce a single unified HTML report with everything.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import log_loss

from . import config
from . import data_loader
from . import evaluate_model
from . import feature_engineering
from . import predict as predict_module
from . import pipeline_progress
from . import preprocessing
from . import report_generator
from . import match_predictor
from . import season_analytics
from . import train_model
from . import visualization_dashboard


def run_pipeline(
    csv_path: Optional[Path] = None,
    *,
    track_progress: bool = False,
) -> None:
    def _p(phase: str, message: str, pct: int) -> None:
        if track_progress:
            pipeline_progress.set_progress(phase, message, pct)

    config.DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. Load player-season data (shared by both pipelines).
    # ==================================================================
    _p("load", "Loading player-season CSV…", 4)
    print("Loading player-season dataset...")
    player_season_df = data_loader.load_player_season_data(csv_path)
    print(f"Loaded {len(player_season_df)} player-season rows.")
    _p("load", f"Loaded {len(player_season_df)} rows", 10)

    # ==================================================================
    # 2. Season-level analytics (Part 1).
    # ==================================================================
    _p("season", "Season analytics (teams, projections, risks)…", 15)
    sa = season_analytics.run_season_analytics(player_season_df)
    _p("season", "Season analytics complete", 22)

    # ==================================================================
    # 3. Synthetic shot generation (xG Part 2).
    # ==================================================================
    _p("shots", "Generating synthetic shot events…", 26)
    print("Generating synthetic shot-level dataset...")
    shots_df = preprocessing.generate_synthetic_shots(player_season_df)
    print(f"Generated {len(shots_df)} synthetic shots.")
    _p("shots", f"{len(shots_df)} synthetic shots", 32)

    # ==================================================================
    # 4. Feature engineering (geometry etc.).
    # ==================================================================
    _p("features", "Engineering shot features…", 36)
    print("Adding geometric features...")
    shots_df = feature_engineering.add_geometric_features(shots_df)

    # ==================================================================
    # 5. Train / test split.
    # ==================================================================
    _p("split", "Train / validation / test split…", 40)
    print("Splitting into train/test sets...")
    train_df, test_df = preprocessing.train_test_split_shots(shots_df)

    internal_train_df, internal_val_df = preprocessing.train_test_split_shots(
        train_df, test_size=0.2
    )

    # ==================================================================
    # 6. Train and tune xG models.
    # ==================================================================
    _p("train", "Training & tuning models (this may take a few minutes)…", 45)
    print("Training and tuning xG models...")
    models = train_model.train_models(internal_train_df, track_progress=track_progress)
    _p("train", "Model tuning complete", 62)

    X_val, y_val = feature_engineering.build_feature_target_matrices(internal_val_df)
    X_test, y_test = feature_engineering.build_feature_target_matrices(test_df)

    # ==================================================================
    # 7. Model selection and calibration.
    # ==================================================================
    _p("select", "Selecting best model & calibrating probabilities…", 68)
    print("Selecting best xG model on validation set...")
    xg_model_comparison = []
    for name, bundle in models.items():
        probs = bundle.pipeline.predict_proba(X_val)[:, 1]
        val_ll = log_loss(y_val, probs)
        xg_model_comparison.append({
            "name": name.replace("_", " ").title(),
            "validation_log_loss": val_ll,
        })

    best_bundle = train_model.select_best_model(models, X_val, y_val)

    print("Calibrating best xG model...")
    calibrated_pipeline = train_model.calibrate_model(best_bundle, X_val, y_val)
    _p("calibrate", "Calibration complete", 74)

    # ==================================================================
    # 8. Final evaluation on held-out test set.
    # ==================================================================
    _p("evaluate", "Evaluating on held-out test set…", 78)
    print("Evaluating calibrated xG model on test set...")
    metrics, cls_report = evaluate_model.evaluate_on_dataframe(calibrated_pipeline, test_df)
    print("=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== Classification Report (threshold=0.5) ===")
    print(cls_report)

    X_test_only, y_test_only = feature_engineering.build_feature_target_matrices(test_df)
    y_proba_test = calibrated_pipeline.predict_proba(X_test_only)[:, 1]

    preprocessor = calibrated_pipeline.named_steps["preprocessor"]
    numeric_features = config.NUMERIC_FEATURES + ["under_pressure"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_feature_names = cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES)
    feature_names = np.concatenate([numeric_features, categorical_feature_names])

    _p("plots", "Creating diagnostic plots…", 84)
    print("Creating diagnostic plots...")
    evaluate_model.plot_diagnostics(
        y_true=y_test_only.values,
        y_proba=y_proba_test,
        feature_names=feature_names,
        model=calibrated_pipeline.named_steps["clf"],
        title_prefix="Calibrated xG Model - ",
        save_dir=config.DATA_OUT_DIR,
    )

    # ==================================================================
    # 9. Save best calibrated xG model.
    # ==================================================================
    _p("save", "Saving model & artefacts…", 90)
    print("Saving calibrated xG pipeline...")
    train_model.save_model(calibrated_pipeline, config.BEST_MODEL_PATH)

    # ==================================================================
    # 10. Sample predictions.
    # ==================================================================
    print("\nExample predictions on first five test shots:")
    sample_shots = test_df.head(5).copy()
    xg = predict_module.predict_xg_for_shots(calibrated_pipeline, sample_shots)
    predict_module.pretty_print_predictions(sample_shots, xg)

    # ==================================================================
    # 11. Save visualization data & generate interactive dashboard.
    # ==================================================================

    dashboard_sample_size = min(15, len(test_df))
    dashboard_sample = test_df.sample(
        dashboard_sample_size, random_state=config.RANDOM_SEED
    ).copy()
    dashboard_xg = predict_module.predict_xg_for_shots(
        calibrated_pipeline, dashboard_sample
    )

    # ==================================================================
    # 12. Build unified HTML report.
    # ==================================================================

    # xG feature importances from the uncalibrated base classifier.
    base_clf = best_bundle.pipeline.named_steps["clf"]
    xg_feat_imp: dict = {}
    if hasattr(base_clf, "feature_importances_"):
        imp = base_clf.feature_importances_
        for i, fn in enumerate(feature_names):
            if i < len(imp):
                xg_feat_imp[str(fn)] = float(imp[i])
    elif hasattr(base_clf, "coef_"):
        coefs = np.abs(np.ravel(base_clf.coef_))
        for i, fn in enumerate(feature_names):
            if i < len(coefs):
                xg_feat_imp[str(fn)] = float(coefs[i])

    # Sample prediction rows for the report.
    sample_preds = []
    for _, row in sample_shots.iterrows():
        idx = len(sample_preds)
        sample_preds.append({
            "player_name": row.get("player_name", "N/A"),
            "x": float(row.get("x", 0)),
            "y": float(row.get("y", 0)),
            "body_part": row.get("body_part", "N/A"),
            "match_minute": row.get("match_minute", "N/A"),
            "xg": float(xg[idx]),
        })

    report_data = report_generator.ReportData(
        # Part 1 — Season analytics
        num_players=sa.num_players,
        num_teams=sa.num_teams,
        total_goals=sa.total_goals,
        total_assists=sa.total_assists,
        team_stats=sa.team_stats,
        top_performers=sa.top_performers,
        goal_model_name=sa.best_model_name,
        goal_model_mae=sa.best_model_mae,
        goal_model_comparison=sa.model_comparison,
        goal_feature_importances=sa.goal_feature_importances,
        player_projections=sa.player_projections,
        team_projections=sa.team_projections,
        discipline_risks=sa.discipline_risks,
        breakout_candidates=sa.breakout_candidates,
        next_season_games=config.NEXT_SEASON_GAMES,
        # Part 2 — xG model
        shots_count=len(shots_df),
        train_size=len(train_df),
        test_size=len(test_df),
        goals_count=int(shots_df["is_goal"].sum()),
        non_goals_count=int((1 - shots_df["is_goal"]).sum()),
        best_model_name=best_bundle.name,
        best_model_params=best_bundle.best_params,
        model_comparison=xg_model_comparison,
        metrics=metrics,
        classification_report=cls_report,
        sample_predictions=sample_preds,
        artifacts_dir=config.DATA_OUT_DIR,
        feature_importances=xg_feat_imp,
    )

    _p("report", "Building unified HTML report…", 93)
    print("\nGenerating unified HTML report...")
    report_generator.generate_html_report(report_data, config.REPORT_PATH)
    print(f"Report saved to: {config.REPORT_PATH}")

    # ==================================================================
    # 13. Interactive visualization dashboard.
    # ==================================================================
    _p("dashboard", "Generating interactive dashboard…", 97)

    dashboard_preds = []
    for i, (_, row) in enumerate(dashboard_sample.iterrows()):
        dashboard_preds.append({
            "player_name": row.get("player_name", "Unknown"),
            "player_position": row.get("player_position", "Unknown"),
            "body_part": row.get("body_part", "Unknown"),
            "x": float(row.get("x", 0)),
            "y": float(row.get("y", 0)),
            "match_minute": int(row.get("match_minute", 0)),
            "xg": float(dashboard_xg[i]),
            "actual": int(row.get("is_goal", 0)),
        })

    dashboard_dataset_info = {
        "total_shots": len(shots_df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "goal_rate": round(float(shots_df["is_goal"].mean()), 4),
    }

    # ==================================================================
    # 13b. Match-level game outcome predictions (1X2) from completed matches.
    # ==================================================================
    match_predictions = []
    match_metrics = {}
    matches_path = config.PROJECT_ROOT / "footy-completed-matches-all-20260330-111727.csv"
    if matches_path.exists():
        try:
            matches_df = match_predictor.load_completed_matches_csv(matches_path)
            match_predictions, match_metrics = match_predictor.build_match_predictions_for_dashboard(
                matches_df, max_rows=18
            )
        except Exception as e:
            print(f"Match prediction model skipped due to error: {e}")

    print("\nSaving visualization JSON files...")
    visualization_dashboard.save_visualization_json(
        metrics=metrics,
        y_true=y_test_only.values,
        y_proba=y_proba_test,
        feature_importances=xg_feat_imp,
        sample_predictions=dashboard_preds,
        match_predictions=match_predictions,
        match_metrics=match_metrics,
        model_comparison=xg_model_comparison,
        model_name=best_bundle.name,
        dataset_info=dashboard_dataset_info,
        output_dir=config.DATA_OUT_DIR,
    )

    print("Generating interactive visualization dashboard...")
    visualization_dashboard.generate_dashboard(
        metrics=metrics,
        y_true=y_test_only.values,
        y_proba=y_proba_test,
        feature_importances=xg_feat_imp,
        sample_predictions=dashboard_preds,
        match_predictions=match_predictions,
        match_metrics=match_metrics,
        model_comparison=xg_model_comparison,
        model_name=best_bundle.name,
        dataset_info=dashboard_dataset_info,
        output_path=config.DASHBOARD_PATH,
        report_data=report_data,
    )


if __name__ == "__main__":
    run_pipeline()
