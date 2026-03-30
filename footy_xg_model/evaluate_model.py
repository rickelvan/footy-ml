"""
Model evaluation utilities for the FootyStats xG system.

This module computes probability‑focused metrics and diagnostic plots:
- Log loss
- ROC‑AUC
- Brier score
- Precision / Recall
- Confusion matrix
- Calibration curve
- Feature importance visualisation
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from . import config
from .feature_engineering import build_feature_target_matrices


def evaluate_probabilistic_predictions(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute scalar probability evaluation metrics.

    Returns a dictionary for easy logging and comparison across models.
    """
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "log_loss": log_loss(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
    }

    # Precision/Recall are threshold‑dependent; we use the default 0.5
    # decision boundary for a quick summary.
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    metrics["precision@0.5"] = precision
    metrics["recall@0.5"] = recall

    return metrics


def plot_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    feature_names: np.ndarray,
    model,
    title_prefix: str = "",
    save_dir=None,
) -> None:
    """
    Generate diagnostic plots for an xG model.

    The plots include:
    - ROC curve.
    - Precision‑Recall curve.
    - Confusion matrix (at threshold 0.5).
    - Calibration curve (predicted vs observed probabilities).
    - Feature importance bar chart.
    """
    if save_dir is None:
        save_dir = config.DATA_OUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- ROC curve ----------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(dpi=config.FIGURE_DPI)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix}ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png")
    plt.close()

    # --- Precision‑Recall curve --------------------------------------------
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(f"{title_prefix}Precision‑Recall Curve")
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curve.png")
    plt.close()

    # --- Confusion matrix ---------------------------------------------------
    y_pred = (y_proba >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        colorbar=False,
        im_kw={"vmin": 0},
        text_kw={"fontsize": 13, "fontweight": "bold"},
    )
    plt.title(f"{title_prefix}Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()

    # --- Calibration curve --------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure(dpi=config.FIGURE_DPI)
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"{title_prefix}Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "calibration_curve.png")
    plt.close()

    # --- Feature importance -------------------------------------------------
    # We try to extract feature importance in a model‑agnostic way:
    # - For tree‑based models we use `feature_importances_`.
    # - For linear models we use absolute coefficients.
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        importances = np.abs(coefs)

    if importances is not None and feature_names is not None:
        # Sort features by importance for readability.
        idx = np.argsort(importances)[::-1][:20]  # top‑20
        plt.figure(dpi=config.FIGURE_DPI, figsize=(8, 6))
        plt.barh(
            np.array(feature_names)[idx][::-1],
            importances[idx][::-1],
        )
        plt.xlabel("Importance")
        plt.title(f"{title_prefix}Top Feature Importances")
        plt.tight_layout()
        plt.savefig(save_dir / "feature_importances.png")
        plt.close()


def evaluate_on_dataframe(pipeline, df: pd.DataFrame) -> Tuple[Dict[str, float], str]:
    """
    Convenience function: evaluate a fitted pipeline on a shots DataFrame.

    Returns metric dictionary and a text classification report.
    """
    X, y = build_feature_target_matrices(df)
    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = evaluate_probabilistic_predictions(y.values, y_proba)

    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y, y_pred)
    return metrics, report

