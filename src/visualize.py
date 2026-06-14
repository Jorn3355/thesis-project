import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MODEL_PATH, REPORTS_DIR, VERDICT_COLORS
from data_gen import generate_match_session
from features import matches_to_dataframe, add_features, FEATURE_COLUMNS
from predict import classify


def diagnostic_figure(player_row, importances, legit_means, triggerbot_means,
                      verdict, prob, top_n=6):
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))


    player_vals     = [player_row[f]      for f in top_features]
    legit_vals      = [legit_means[f]      for f in top_features]
    triggerbot_vals = [triggerbot_means[f] for f in top_features]

    x = np.arange(len(top_features))
    w = 0.27

    axes[0].bar(x - w, legit_vals,      w, label="legit mean",      color="#3498db", alpha=0.85)
    axes[0].bar(x,     player_vals,     w, label="this player",     color=VERDICT_COLORS[verdict])
    axes[0].bar(x + w, triggerbot_vals, w, label="triggerbot mean", color="#e74c3c", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(top_features, rotation=30, ha="right")
    axes[0].set_title(f"Player vs population (top {top_n} features)")
    axes[0].legend(loc="best")
    axes[0].grid(axis="y", alpha=0.3)


    top10 = importances.sort_values(ascending=True).tail(10)
    axes[1].barh(top10.index, top10.values, color="#2c3e50")
    axes[1].set_title("Model feature importances (top 10)")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Verdict: {verdict}    triggerbot probability: {prob:.3f}",
        fontsize=14,
        color=VERDICT_COLORS[verdict],
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def main():
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print("Generating reference distribution + test batch...")
    reference = [generate_match_session() for _ in range(2000)]
    test      = [generate_match_session() for _ in range(5)]

    ref_df  = add_features(matches_to_dataframe(reference))
    test_df = add_features(matches_to_dataframe(test))
    X_test  = test_df[FEATURE_COLUMNS]

    results = classify(model, X_test)

    legit_means      = ref_df[ref_df["is_cheating"] == 0][FEATURE_COLUMNS].mean()
    triggerbot_means = ref_df[ref_df["is_cheating"] == 1][FEATURE_COLUMNS].mean()
    importances      = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(len(X_test)):
        row     = test_df.iloc[i]
        prob    = results.iloc[i]["triggerbot_probability"]
        verdict = results.iloc[i]["verdict"]

        fig = diagnostic_figure(
            row, importances, legit_means, triggerbot_means, verdict, prob,
        )
        out = REPORTS_DIR / f"diagnostic_{row['puuid']}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        actual = "triggerbot" if row["is_cheating"] == 1 else "legit"
        print(f"  {out.name}   verdict={verdict:10s}  prob={prob:.3f}  (actual={actual})")

    print(f"\nReports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
