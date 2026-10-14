# model.py
# Phase 5: here I train a Random Forest, choose the decision threshold fairly,
# and save the final model for phases 6 and 8.
#
# I run it from the project folder with:
#     python -m laser_defect.model
#
# Why a Random Forest? It combines many decision trees, so it can learn curved,
# non-linear rules (e.g. "unsound if contrast is low AND the band moved") that
# logistic regression's single straight line cannot.
#
# Why choose a threshold? The model gives a probability of "Unsound". By default
# I would say "Unsound" above 50%, but another cut-off may balance missed defects
# and false alarms better. I must choose it WITHOUT looking at the test welds,
# otherwise my test score would be too optimistic.

from pathlib import Path

import joblib                     # I use joblib to save the trained model to a file
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from laser_defect.features import FEATURE_COLUMNS, load_features

MODEL_FILE = Path("models/defect_classifier.joblib")
RESULTS_FILE = Path("reports/phase5_random_forest.csv")
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2)   # cut-offs I try: 0.20, 0.25, ... 0.80


def make_forest(seed=0):
    # 200 trees. min_samples_leaf=20: every final leaf must contain at least 20
    # frames, which stops trees from memorising single noisy frames (overfitting).
    # random_state fixes the randomness, so I get the same result every run.
    # n_jobs=-1 uses all my computer's cores to train faster.
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)


def out_of_fold_probabilities(X, y, groups, n_folds):
    # For every frame I get a prediction from a model that never saw its weld.
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        prob[te] = make_forest().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return prob


def best_threshold(y, prob):
    # I try every cut-off and keep the one with the highest balanced accuracy.
    scores = [balanced_accuracy_score(y, (prob >= t).astype(int)) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def evaluate(df, columns=FEATURE_COLUMNS, n_folds=5, inner_folds=3):
    # "Nested" cross-validation:
    #   OUTER loop: 5 rounds, each tests on welds the model never saw.
    #   INNER loop: inside the training welds only, I pick the threshold.
    X, y, groups = df[list(columns)].to_numpy(), df["y"].to_numpy(), df["weld"].to_numpy()
    rows, total_cm = [], np.zeros((2, 2), dtype=int)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])          # no weld in both
        inner_prob = out_of_fold_probabilities(X[tr], y[tr], groups[tr], inner_folds)
        threshold = best_threshold(y[tr], inner_prob)         # chosen on training welds only
        prob = make_forest().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        pred = (prob >= threshold).astype(int)
        total_cm += confusion_matrix(y[te], pred, labels=[0, 1])
        rows.append({"threshold": threshold,
                     "roc_auc": roc_auc_score(y[te], prob),
                     "balanced_accuracy": balanced_accuracy_score(y[te], pred),
                     "f1_unsound": f1_score(y[te], pred)})
    return pd.DataFrame(rows), total_cm


def train_final(df, columns=FEATURE_COLUMNS, n_folds=5):
    # After evaluating, I train ONE final model on all welds, with a threshold
    # chosen by weld-based cross-validation on all data, and save both together.
    X, y, groups = df[list(columns)].to_numpy(), df["y"].to_numpy(), df["weld"].to_numpy()
    threshold = best_threshold(y, out_of_fold_probabilities(X, y, groups, n_folds))
    model = make_forest().fit(X, y)
    return {"model": model, "threshold": threshold, "features": list(columns)}


# This block runs only when I start the file directly: python -m laser_defect.model
if __name__ == "__main__":
    df = load_features()
    print("Nested weld-grouped cross-validation (takes about a minute)...")
    scores, cm = evaluate(df)
    print(scores.round(3).to_string(index=False))
    print("\nMean ± std:")
    for m in ("roc_auc", "balanced_accuracy", "f1_unsound"):
        print(f"  {m:18s} {scores[m].mean():.3f} ± {scores[m].std():.3f}")
    # Confusion matrix: rows = truth, columns = prediction, summed over all test welds.
    print("\nConfusion matrix (all test welds):")
    print(pd.DataFrame(cm, index=["true Sound", "true Unsound"],
                       columns=["pred Sound", "pred Unsound"]))
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_FILE, index=False)

    bundle = train_final(df)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nSaved {RESULTS_FILE} and {MODEL_FILE} (threshold {bundle['threshold']})")
