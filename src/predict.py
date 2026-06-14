import joblib
import pandas as pd

from config import MODEL_PATH, PASS_THRESHOLD, TRIGGERBOT_THRESHOLD
from data_gen import generate_match_session
from features import matches_to_dataframe, add_features, FEATURE_COLUMNS


def verdict_from_prob(prob: float) -> str:
    if prob < PASS_THRESHOLD:
        return "PASS"
    if prob < TRIGGERBOT_THRESHOLD:
        return "REVIEW"
    return "TRIGGERBOT"


def classify(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return per-row triggerbot probability + verdict."""
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "triggerbot_probability": proba,
        "verdict": [verdict_from_prob(p) for p in proba],
    })

def explain_row(row, importances, legit_means, triggerbot_means, top_n=5):
    """Print the top-N model features for this row, alongside class means."""
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()
    print("  Top contributing features:")
    for feat in top_features:
        val = row[feat]
        legit  = legit_means[feat]
        trig   = triggerbot_means[feat]
        closer = "triggerbot" if abs(val - trig) < abs(val - legit) else "legit"
        print(f"    {feat:22s}  value={val:8.2f}  "
              f"legit~{legit:7.2f}  triggerbot~{trig:7.2f}  -> closer to {closer}")


def main():
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)



    print("Generating reference distribution + test batch...")
    reference = [generate_match_session() for _ in range(2000)]
    test      = [generate_match_session() for _ in range(10)]

    ref_df  = add_features(matches_to_dataframe(reference))
    test_df = add_features(matches_to_dataframe(test))
    X_test  = test_df[FEATURE_COLUMNS]

    results = classify(model, X_test)

    legit_means      = ref_df[ref_df["is_cheating"] == 0][FEATURE_COLUMNS].mean()
    triggerbot_means = ref_df[ref_df["is_cheating"] == 1][FEATURE_COLUMNS].mean()
    importances      = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)

    print("\n" + "=" * 80)
    correct = 0
    for i in range(len(X_test)):
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

    print("\n" + "=" * 80)
    print(f"Verdict matched ground truth: {correct}/{len(X_test)}")


if __name__ == "__main__":
    main()
