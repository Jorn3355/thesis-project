"""Feature-distribution statistical tests with effect sizes.

For each key feature, compares the cheater vs legit distributions using:
  - Mann-Whitney U (non-parametric — appropriate for clipped Gaussians where
    the t-test's normality assumption is violated)
  - Rank-biserial correlation (effect size native to Mann-Whitney U)
  - Cohen's d (standardized mean difference — conventional baseline)

Why the effect size matters: at n >= 5000, even tiny mean differences produce
p < 0.001, so the p-value alone is not informative. Rank-biserial and Cohen's d
quantify how *separable* the classes actually are.

Effect size conventions (absolute value):
  Cohen's d           small=0.2  medium=0.5  large=0.8  very_large>1.0
  Rank-biserial r     small=0.1  medium=0.3  large=0.5
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu




KEY_FEATURES = [
    "time_on_target",
    "reaction_time",
    "aim_deviation",
    "hit_accuracy",
    "shots_fired",
    "precision_score",
    "score_z",
    "kills_z",
    "tier_relative_score",
    "tier_relative_kills",
]


def cohens_d(legit_vals: np.ndarray, cheat_vals: np.ndarray) -> float:
    """Standardized mean difference. Positive = cheater higher than legit."""
    pooled_var = (np.var(legit_vals, ddof=1) + np.var(cheat_vals, ddof=1)) / 2
    if pooled_var == 0:
        return 0.0
    return float((np.mean(cheat_vals) - np.mean(legit_vals)) / np.sqrt(pooled_var))


def rank_biserial(u_stat_for_legit: float, n_legit: int, n_cheat: int) -> float:
    """Rank-biserial correlation effect size for Mann-Whitney U.

    Sign convention with U computed on legit:
      r > 0  means cheater values rank higher than legit
      r < 0  means cheater values rank lower than legit
      |r|=1  perfect separation, |r|=0  no separation
    """
    return float(1 - (2 * u_stat_for_legit) / (n_legit * n_cheat))


def feature_distribution_tests(
    df: pd.DataFrame,
    features=None,
    label_col: str = "is_cheating",
) -> pd.DataFrame:
    """Run Mann-Whitney U + effect sizes for each feature in `features`.

    Returns a DataFrame indexed by feature, sorted by |rank_biserial_r|
    (most discriminative features first).
    """
    if features is None:
        features = KEY_FEATURES

    legit_mask = df[label_col] == 0
    cheat_mask = df[label_col] == 1
    n_legit = int(legit_mask.sum())
    n_cheat = int(cheat_mask.sum())

    rows = []
    for feat in features:
        legit_vals = df.loc[legit_mask, feat].values
        cheat_vals = df.loc[cheat_mask, feat].values

        u_stat, p_val = mannwhitneyu(legit_vals, cheat_vals, alternative="two-sided")
        rb = rank_biserial(u_stat, n_legit, n_cheat)
        d  = cohens_d(legit_vals, cheat_vals)

        rows.append({
            "feature":          feat,
            "mean_legit":       float(np.mean(legit_vals)),
            "mean_triggerbot":  float(np.mean(cheat_vals)),
            "cohens_d":         d,
            "rank_biserial_r":  rb,
            "U_stat":           float(u_stat),
            "p_value":          float(p_val),
        })

    out = pd.DataFrame(rows).set_index("feature")
    out = out.reindex(out["rank_biserial_r"].abs().sort_values(ascending=False).index)
    return out


def plot_effect_sizes(results: pd.DataFrame, save_path: Path) -> None:
    """Two-panel horizontal bar chart: Cohen's d and rank-biserial r per feature.

    Bars are color-coded by direction (red = triggerbot higher, blue = lower).
    Dotted reference lines mark the small / medium / large effect-size conventions.
    """


    d_sorted = results.reindex(results["cohens_d"].abs().sort_values(ascending=True).index)
    r_sorted = results.reindex(results["rank_biserial_r"].abs().sort_values(ascending=True).index)

    d_colors = ["#e74c3c" if v > 0 else "#3498db" for v in d_sorted["cohens_d"]]
    r_colors = ["#e74c3c" if v > 0 else "#3498db" for v in r_sorted["rank_biserial_r"]]

    fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=(15, 7))


    ax_d.barh(d_sorted.index, d_sorted["cohens_d"].values, color=d_colors)
    ax_d.axvline(0, color="black", linewidth=0.6)
    for thresh, label in [(0.2, "small"), (0.5, "med"), (0.8, "large")]:
        for sign in (1, -1):
            ax_d.axvline(sign * thresh, color="gray", linestyle=":", alpha=0.5)
        ax_d.text(thresh, len(d_sorted) - 0.4, label, fontsize=8, color="gray", ha="center")
    ax_d.set_xlabel("Cohen's d  (negative = triggerbot lower, positive = triggerbot higher)")
    ax_d.set_title("Cohen's d effect size per feature")
    ax_d.grid(axis="x", alpha=0.3)


    ax_r.barh(r_sorted.index, r_sorted["rank_biserial_r"].values, color=r_colors)
    ax_r.axvline(0, color="black", linewidth=0.6)
    for thresh, label in [(0.1, "small"), (0.3, "med"), (0.5, "large")]:
        for sign in (1, -1):
            ax_r.axvline(sign * thresh, color="gray", linestyle=":", alpha=0.5)
        ax_r.text(thresh, len(r_sorted) - 0.4, label, fontsize=8, color="gray", ha="center")
    ax_r.set_xlabel("Rank-biserial r  (Mann-Whitney U effect size)")
    ax_r.set_xlim(-1.05, 1.05)
    ax_r.set_title("Rank-biserial r effect size per feature")
    ax_r.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature-distribution effect sizes  (legit vs. triggerbot)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distribution_overlap(
    df: pd.DataFrame,
    feature: str,
    save_path: Path,
    title: str = None,
    bins: int = 60,
    label_col: str = "is_cheating",
) -> None:
    """Overlaid histograms of `feature` for legit vs. triggerbot.

    Generic helper — call once per feature you want a distribution figure for.
    Shows class means as dashed vertical lines so the gap (and tail overlap)
    is visually obvious.
    """
    legit_vals = df.loc[df[label_col] == 0, feature].values
    cheat_vals = df.loc[df[label_col] == 1, feature].values


    all_vals = np.concatenate([legit_vals, cheat_vals])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(legit_vals, bins=bin_edges, density=True, alpha=0.55,
            color="#3498db", edgecolor="white", linewidth=0.3,
            label=f"legit  (n={len(legit_vals)})")
    ax.hist(cheat_vals, bins=bin_edges, density=True, alpha=0.55,
            color="#e74c3c", edgecolor="white", linewidth=0.3,
            label=f"triggerbot  (n={len(cheat_vals)})")

    legit_mean = float(np.mean(legit_vals))
    cheat_mean = float(np.mean(cheat_vals))
    ax.axvline(legit_mean, color="#1f6aa5", linestyle="--", linewidth=2, alpha=0.85,
               label=f"legit mean = {legit_mean:.2f}")
    ax.axvline(cheat_mean, color="#a93226", linestyle="--", linewidth=2, alpha=0.85,
               label=f"triggerbot mean = {cheat_mean:.2f}")

    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution of {feature}: legit vs. triggerbot")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
