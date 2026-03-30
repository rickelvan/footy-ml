"""
Model training module for the FootyStats xG system.

Responsibilities:
- Define ML pipelines for multiple algorithms (logistic regression, random forest,
  gradient boosting).
- Perform cross‑validated hyper‑parameter tuning.
- Select the best performing model based on log loss.
- Optionally calibrate predicted probabilities to improve probabilistic accuracy.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from . import pipeline_progress
from .feature_engineering import build_feature_target_matrices


@dataclass
class TrainedModelBundle:
    """
    Container for a trained model and metadata.

    Storing the preprocessing and model together simplifies downstream
    prediction and evaluation.
    """

    name: str
    pipeline: Pipeline
    best_params: Dict
    cv_results: pd.DataFrame


def _build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Standardises numeric features.
    - One‑hot encodes categorical features.
    """
    numeric_features = config.NUMERIC_FEATURES + ["under_pressure"]
    categorical_features = config.CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def _build_model_grids():
    """
    Define model pipelines and their corresponding hyper‑parameter grids.

    Each entry returned is (model_name, pipeline, param_grid).
    """
    preprocessor = _build_preprocessor()

    pipelines_and_grids = []

    # --- Logistic Regression (baseline) -------------------------------------
    log_reg = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=None,
        random_state=config.RANDOM_SEED,
    )
    log_reg_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", log_reg),
        ]
    )
    log_reg_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }
    pipelines_and_grids.append(("logistic_regression", log_reg_pipe, log_reg_grid))

    # --- Random Forest ------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", rf),
        ]
    )
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 5],
    }
    pipelines_and_grids.append(("random_forest", rf_pipe, rf_grid))

    # --- Gradient Boosting --------------------------------------------------
    gb = GradientBoostingClassifier(random_state=config.RANDOM_SEED)
    gb_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", gb),
        ]
    )
    gb_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
    }
    pipelines_and_grids.append(("gradient_boosting", gb_pipe, gb_grid))

    return pipelines_and_grids


def train_models(
    train_df: pd.DataFrame,
    *,
    track_progress: bool = False,
) -> Dict[str, TrainedModelBundle]:
    """
    Train and tune multiple xG models using cross‑validated log loss.

    Parameters
    ----------
    train_df:
        Training shots DataFrame after feature engineering.

    Returns
    -------
    models : dict
        Mapping from model name to `TrainedModelBundle`.
    """
    X_train, y_train = build_feature_target_matrices(train_df)

    # Use negative log loss as the optimisation objective; this is standard
    # for probabilistic models where calibrated probabilities matter.
    log_loss_scorer = make_scorer(
        log_loss, response_method="predict_proba", greater_is_better=False
    )

    results: Dict[str, TrainedModelBundle] = {}
    grids = _build_model_grids()
    n_grids = len(grids)
    for gi, (name, pipeline, param_grid) in enumerate(grids):
        if track_progress:
            pct = 45 + int((gi / max(n_grids, 1)) * 14)
            pipeline_progress.set_progress(
                "train",
                f"Tuning {name.replace('_', ' ')} (this step is slow)…",
                min(58, pct),
            )
        # n_jobs=1 avoids nested joblib + RF n_jobs=-1 oversubscription, which can
        # hang or thrash on Windows (especially when training runs in a Flask thread).
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=log_loss_scorer,
            cv=config.CV_FOLDS,
            n_jobs=1,
        )
        grid.fit(X_train, y_train)

        cv_results_df = pd.DataFrame(grid.cv_results_)
        results[name] = TrainedModelBundle(
            name=name,
            pipeline=grid.best_estimator_,
            best_params=grid.best_params_,
            cv_results=cv_results_df,
        )

    return results


def select_best_model(models: Dict[str, TrainedModelBundle], X_valid: pd.DataFrame, y_valid: pd.Series) -> TrainedModelBundle:
    """
    Select the best model based on validation log loss.
    """
    best_name = None
    best_bundle = None
    best_logloss = np.inf

    for name, bundle in models.items():
        probs = bundle.pipeline.predict_proba(X_valid)[:, 1]
        ll = log_loss(y_valid, probs)
        if ll < best_logloss:
            best_logloss = ll
            best_name = name
            best_bundle = bundle

    if best_bundle is None:
        raise RuntimeError("No models provided for selection.")

    print(f"Best model on validation log loss: {best_name} (log loss={best_logloss:.4f})")
    return best_bundle


def calibrate_model(bundle: TrainedModelBundle, X_valid: pd.DataFrame, y_valid: pd.Series) -> Pipeline:
    """
    Wrap the classifier in a probability calibration layer.

    We apply isotonic or sigmoid calibration via `CalibratedClassifierCV` on
    held‑out validation data, which generally improves the alignment between
    predicted probabilities and observed frequencies (critical for xG).

    Note: we only calibrate the final classifier, not the preprocessing stage.
    """
    # Extract the trained preprocessing and classifier objects.
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    clf = bundle.pipeline.named_steps["clf"]

    # Transform validation features once using the preprocessor.
    X_valid_trans = preprocessor.transform(X_valid)

    # Use 'estimator' (sklearn 1.2+); older versions used 'base_estimator'.
    calibrator = CalibratedClassifierCV(
        estimator=clf,
        method="isotonic",
        cv="prefit",  # we already trained `clf`; no need to refit.
    )
    calibrator.fit(X_valid_trans, y_valid)

    # Rebuild a pipeline that chains the existing preprocessor
    # with the calibrated classifier.
    calibrated_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", calibrator),
        ]
    )
    return calibrated_pipeline


def save_model(pipeline: Pipeline, path=None) -> None:
    """
    Persist a trained pipeline (preprocessing + classifier) to disk.
    """
    if path is None:
        path = config.BEST_MODEL_PATH
    path = config.Path(path) if hasattr(config, "Path") else path  # defensive

    config.MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved calibrated xG pipeline to {path}")

