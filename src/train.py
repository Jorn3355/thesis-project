import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import MODEL_PATH, N_TRAIN, RANDOM_STATE, TEST_SIZE
from data_gen import generate_match_session
from features import FEATURE_COLUMNS, add_features, matches_to_dataframe
from validation import (
    auc_metrics, make_default_model, print_cv_results,
    run_cross_validation, seed_everything,
)


def main():
    seed_everything()


    print(f"Generating {N_TRAIN} matches...")
    matches = [generate_match_session() for _ in range(N_TRAIN)]
    df = add_features(matches_to_dataframe(matches))

    X = df[FEATURE_COLUMNS]
    y = df["is_cheating"]

    print(f"  rows={X.shape[0]}  features={X.shape[1]}  "
          f"triggerbots={int(y.sum())}  legits={int((y == 0).sum())}\n")


    print("=== 5-fold cross-validation ===")
    cv_results = run_cross_validation(X, y, n_splits=5)
    print_cv_results(cv_results, n_splits=5)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print("\nTraining final RandomForestClassifier on train split...")
    model = make_default_model()
    model.fit(X_train, y_train)


    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification report (held-out test set) ===")
    print(classification_report(
        y_test, y_pred,
        target_names=["legit", "triggerbot"],
        digits=3,
    ))

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["actual_legit", "actual_triggerbot"],
        columns=["pred_legit", "pred_triggerbot"],
    ))

    aucs = auc_metrics(y_test, y_proba)
    print(f"\nROC-AUC (held-out test): {aucs['roc_auc']:.4f}")
    print(f"PR-AUC  (held-out test): {aucs['pr_auc']:.4f}")


    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\n=== Top 10 feature importances ===")
    print(importances.sort_values(ascending=False).head(10).round(4))


    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
