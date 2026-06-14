import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import (
    MODEL_PATH, REPORTS_DIR,
    N_TRAIN, N_TEST, TEST_SIZE, RANDOM_STATE,
)
from data_gen import generate_match_session
from features import matches_to_dataframe, add_features, FEATURE_COLUMNS
from predict import classify, explain_row
from robustness import plot_robustness_curve, run_robustness_sweep
from stats import (
    feature_distribution_tests, plot_effect_sizes,
    plot_feature_distribution_overlap,
)
from validation import (
    auc_metrics, compare_models, comparison_dataframe,
    make_default_model, plot_feature_importances, plot_roc_pr_curves,
    seed_everything,
)
from visualize import diagnostic_figure


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(f" {text}")
    print("=" * 78)


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 9999)
    seed_everything()




    banner("PHASE 1 + 2  -  Data generation + feature engineering")
    print(f"Generating {N_TRAIN} match sessions...")
    matches = [generate_match_session() for _ in range(N_TRAIN)]
    df = add_features(matches_to_dataframe(matches))
    X = df[FEATURE_COLUMNS]
    y = df["is_cheating"]
    print(f"  rows={len(X)}  features={X.shape[1]}  "
          f"triggerbots={int(y.sum())}  legits={int((y == 0).sum())}")







    banner("PHASE 2b  -  Statistical feature analysis (Mann-Whitney U + effect sizes)")
    stats_table = feature_distribution_tests(df)
    print(stats_table.round(4).to_string())
    print(
        "\nNote: rows sorted by |rank_biserial_r| (most discriminative first).\n"
        "Convention: cohens_d / rank_biserial_r are positive when cheater > legit, negative when cheater < legit.\n"
        "Magnitude guide: |d|>=0.8 large, |r|>=0.5 large, ~0 means no class separation."
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    effect_sizes_path = REPORTS_DIR / "effect_sizes.png"
    plot_effect_sizes(stats_table, effect_sizes_path)
    print(f"\nEffect-size plot saved -> {effect_sizes_path}")




    tot_path = REPORTS_DIR / "time_on_target_distributions.png"
    plot_feature_distribution_overlap(
        df, "time_on_target", tot_path,
        title="Distribution of time_on_target: legit vs. triggerbot",
    )
    print(f"time_on_target distribution plot saved -> {tot_path}")





    banner("PHASE 3a  -  Multi-model 5-fold cross-validation")
    comparison = compare_models(X, y, n_splits=5)
    print(comparison_dataframe(comparison).to_string())




    banner("PHASE 3b  -  Final model train + held-out evaluation")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    model = make_default_model()
    model.fit(X_tr, y_tr)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    print("\nClassification report (held-out test set):")
    print(classification_report(
        y_te, y_pred, target_names=["legit", "triggerbot"], digits=3,
    ))
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_te, y_pred),
        index=["actual_legit", "actual_triggerbot"],
        columns=["pred_legit", "pred_triggerbot"],
    ))

    aucs = auc_metrics(y_te, y_proba)
    print(f"\nROC-AUC (held-out test): {aucs['roc_auc']:.4f}")
    print(f"PR-AUC  (held-out test): {aucs['pr_auc']:.4f}")


    roc_pr_path = REPORTS_DIR / "roc_pr_curves.png"
    plot_roc_pr_curves(X_tr, y_tr, X_te, y_te, roc_pr_path)
    print(f"ROC + PR curves saved -> {roc_pr_path}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop 10 feature importances:")
    print(importances.sort_values(ascending=False).head(10).round(4))

    fi_path = REPORTS_DIR / "feature_importances.png"
    plot_feature_importances(importances, fi_path)
    print(f"Feature importances plot saved -> {fi_path}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")







    banner("PHASE 3c  -  Subtle-triggerbot robustness sweep (model held fixed)")
    sweep_results = run_robustness_sweep(model, n_matches_per_level=2000)
    print(sweep_results.round(4).to_string())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_plot_path = REPORTS_DIR / "robustness_curve.png"
    plot_robustness_curve(sweep_results, sweep_plot_path)
    print(f"\nRobustness curve saved -> {sweep_plot_path}")

    legit_means      = df[df["is_cheating"] == 0][FEATURE_COLUMNS].mean()
    triggerbot_means = df[df["is_cheating"] == 1][FEATURE_COLUMNS].mean()

    banner(f"PHASE 4  -  Predict + explain on {N_TEST} fresh players")
    test_matches = [generate_match_session() for _ in range(N_TEST)]
    test_df = add_features(matches_to_dataframe(test_matches))
    results = classify(model, test_df[FEATURE_COLUMNS])

    correct = 0
    for i in range(len(test_df)):
        row     = test_df.iloc[i]
        prob    = results.iloc[i]["triggerbot_probability"]
        verdict = results.iloc[i]["verdict"]
        actual  = "triggerbot" if row["is_cheating"] == 1 else "legit"

        match = (verdict == "TRIGGERBOT" and actual == "triggerbot") or \
                (verdict == "PASS"       and actual == "legit")      or \
                (verdict == "REVIEW")
        if match:
            correct += 1

        print(f"\nPlayer {row['puuid']}  tier={row['tier']}  actual={actual}")
        print(f"  Verdict: {verdict:10s}  triggerbot probability: {prob:.3f}")
        explain_row(row, importances, legit_means, triggerbot_means)

    print(f"\nVerdicts matching ground truth: {correct}/{len(test_df)}")




    banner(f"PHASE 5  -  Render diagnostic reports -> {REPORTS_DIR}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(len(test_df)):
        row     = test_df.iloc[i]
        prob    = results.iloc[i]["triggerbot_probability"]
        verdict = results.iloc[i]["verdict"]

        fig = diagnostic_figure(
            row, importances, legit_means, triggerbot_means, verdict, prob,
        )
        out = REPORTS_DIR / f"diagnostic_{row['puuid']}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")

    banner("DONE")


if __name__ == "__main__":
    main()
