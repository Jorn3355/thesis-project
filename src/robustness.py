"""Subtle-triggerbot robustness sweep.

Trains nothing — takes an already-trained model and evaluates it against
held-out test sets where the triggerbot's hold_time distribution is shifted
toward the legit range. Produces a degradation curve answering:

  "Does the model break gracefully as the cheat becomes more human-like,
   or does detection collapse to near-chance?"

The cheater's OTHER signals (shots_fired, accuracy, score inflation) stay at
their default values. Only hold_time varies — so the curve isolates how much
detection depends on per-event timing vs. tier-relative player aggregates.

Legit hold_time mean is ~280 ms. Sweep ends at 240 ms (cheater_hold_time
distribution overlaps the legit distribution almost completely).
"""
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from data_gen import generate_match_session
from features import FEATURE_COLUMNS, add_features, matches_to_dataframe



SUBTLETY_LEVELS: List[Tuple[str, float, float]] = [
    ("blatant",          40,  20),
    ("moderate",         80,  30),
    ("subtle",          120,  40),
    ("very_subtle",     180,  50),
    ("near_human",      240,  60),
]


def run_robustness_sweep(model, n_matches_per_level: int = 2000) -> pd.DataFrame:
    """Evaluate `model` on a fresh test set at each subtlety level."""
    rows = []
    for label, mean, std in SUBTLETY_LEVELS:
        matches = [
            generate_match_session(
                cheater_hold_time_mean=mean,
                cheater_hold_time_std=std,
            )
            for _ in range(n_matches_per_level)
        ]
        df = add_features(matches_to_dataframe(matches))
        X = df[FEATURE_COLUMNS]
        y = df["is_cheating"]

        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        rows.append({
            "subtlety":           label,
            "cheater_hold_mean":  mean,
            "cheater_hold_std":   std,
            "accuracy":           accuracy_score(y, y_pred),
            "precision":          precision_score(y, y_pred, zero_division=0),
            "recall":             recall_score(y, y_pred, zero_division=0),
            "f1":                 f1_score(y, y_pred, zero_division=0),
            "roc_auc":            roc_auc_score(y, y_proba),
            "pr_auc":             average_precision_score(y, y_proba),
        })
    return pd.DataFrame(rows).set_index("subtlety")


def plot_robustness_curve(results: pd.DataFrame, save_path: Path) -> None:
    """Render the degradation curve as a single PNG."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_to_plot = [
        ("accuracy",  "#3498db"),
        ("precision", "#e74c3c"),
        ("recall",    "#2ecc71"),
        ("f1",        "#9b59b6"),
        ("roc_auc",   "#f39c12"),
    ]
    x = results["cheater_hold_mean"].values

    for metric, color in metrics_to_plot:
        ax.plot(x, results[metric].values, marker="o", label=metric,
                color=color, linewidth=2, markersize=7)


    ax.axvline(280, color="gray", linestyle="--", alpha=0.6,
               label="legit mean hold_time (~280 ms)")


    for label, mean, _ in SUBTLETY_LEVELS:
        ax.annotate(label, xy=(mean, 0.02), xytext=(mean, -0.03),
                    ha="center", va="top", fontsize=8, color="gray",
                    annotation_clip=False)

    ax.set_xlabel("Triggerbot hold_time mean (ms)  —  higher = more subtle, approaching legit")
    ax.set_ylabel("Score")
    ax.set_title("Robustness sweep: detection degradation as triggerbot timing becomes human-like")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(x.min() - 20, max(x.max(), 280) + 20)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
