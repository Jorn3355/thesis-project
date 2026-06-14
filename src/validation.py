"""Validation helpers — k-fold cross-validation and discrimination metrics.

Used by main.py and train.py so the same CV protocol is applied consistently.
"""
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, RF_MAX_DEPTH, RF_N_ESTIMATORS

CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed Python's `random` and numpy. Call once at the top of every entry point."""
    random.seed(seed)
    np.random.seed(seed)


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )


def make_logistic_regression() -> Pipeline:
    """Logistic regression wrapped in a StandardScaler — linear models need scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def candidate_models() -> Dict[str, object]:
    """The model family lineup used by compare_models()."""
    return {
        "RandomForest":       make_random_forest(),
        "GradientBoosting":   make_gradient_boosting(),
        "LogisticRegression": make_logistic_regression(),
    }


def make_default_model() -> RandomForestClassifier:
    """Backward-compat alias — the saved/deployed model is still Random Forest."""
    return make_random_forest()


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Stratified k-fold CV. Returns {metric: (mean, std)} for CV_METRICS."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model = make_default_model()
    results = cross_validate(model, X, y, cv=cv, scoring=CV_METRICS, n_jobs=-1)
    return {
        m: (float(results[f"test_{m}"].mean()), float(results[f"test_{m}"].std()))
        for m in CV_METRICS
    }


def print_cv_results(results: Dict[str, Tuple[float, float]], n_splits: int = 5) -> None:
    print(f"Cross-validation metrics ({n_splits}-fold, mean +/- std):")
    for metric, (mean, std) in results.items():
        print(f"  {metric:20s}  {mean:.4f} +/- {std:.4f}")


def auc_metrics(y_true, y_proba) -> Dict[str, float]:
    """Compute ROC-AUC and PR-AUC on the held-out test set."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc":  float(average_precision_score(y_true, y_proba)),
    }


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Run each candidate model through k-fold CV under identical conditions.

    Returns {model_name: {metric: (mean, std)}} — same structure as
    run_cross_validation() but keyed by model.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, model in candidate_models().items():
        results = cross_validate(model, X, y, cv=cv, scoring=CV_METRICS, n_jobs=-1)
        out[name] = {
            m: (float(results[f"test_{m}"].mean()), float(results[f"test_{m}"].std()))
            for m in CV_METRICS
        }
    return out


def comparison_dataframe(
    comparison: Dict[str, Dict[str, Tuple[float, float]]],
) -> pd.DataFrame:
    """Format compare_models() output as a pretty 'mean +/- std' table."""
    rows = []
    for name, metrics in comparison.items():
        row = {"model": name}
        for metric, (mean, std) in metrics.items():
            row[metric] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_roc_pr_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    save_path: Path,
) -> None:
    """ROC + Precision-Recall curves for each candidate model.

    Fits each candidate model fresh on `X_train, y_train`, then plots both
    curves against `X_test, y_test`. AUCs shown in legends.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["#2c3e50", "#e74c3c", "#3498db"]

    for (name, model), color in zip(candidate_models().items(), colors):
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]


        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name}  AUC={roc_auc:.4f}")


        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        ax_pr.plot(rec, prec, color=color, linewidth=2,
                   label=f"{name}  AP={pr_auc:.4f}")


    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance (AUC=0.5)")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves (held-out test set)")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)


    pos_rate = float((y_test == 1).mean())
    ax_pr.axhline(pos_rate, color="black", linestyle="--", alpha=0.5,
                  label=f"baseline = {pos_rate:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curves (held-out test set)")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.legend(loc="lower left")
    ax_pr.grid(alpha=0.3)

    fig.suptitle("Discrimination curves across candidate models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(
    importances: pd.Series,
    save_path: Path,
    title: str = "Random Forest feature importances (Gini-based)",
) -> None:
    """Horizontal bar chart of every feature's importance, sorted descending.

    All features are shown — including dead-weight ones (e.g. wallbang) — so
    the visual makes it obvious which features the model leans on and which
    it ignores.
    """
    sorted_imp = importances.sort_values(ascending=True)
    n = len(sorted_imp)


    colors = plt.cm.viridis(np.linspace(0.20, 0.85, n))

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.32)))
    bars = ax.barh(sorted_imp.index, sorted_imp.values, color=colors)


    max_val = float(sorted_imp.max())
    pad = max_val * 0.01 if max_val > 0 else 0.001
    for bar, val in zip(bars, sorted_imp.values):
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#333")

    ax.set_xlabel("Importance")
    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
