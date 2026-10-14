# train.py
# Phase 4: here I train my first, simple models and judge them fairly.
#
# I run it from the project folder with:
#     python -m laser_defect.train
#
# Two ideas make the evaluation honest:
#   1. SPLIT BY WELD. Frames of one weld look very similar. If some frames of a
#      weld were in training and others in testing, the model would "recognise"
#      the weld instead of learning what a defect looks like. So every weld is
#      either fully in training or fully in testing (GroupKFold does this).
#   2. A "CHEATING" BASELINE. Unsound frames usually come late in a weld (phase 2).
#      I train one model on POSITION ONLY. My thermal model must be judged against
#      it: if it is not clearly better, it learned time, not heat.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from laser_defect.features import FEATURE_COLUMNS, load_features

RESULTS_FILE = Path("reports/phase4_baselines.csv")
N_FOLDS = 5          # 5 rounds; each round tests on a different 1/5 of the welds


def add_position(df):
    # Position inside the weld: 0 = first labelled frame, 1 = last.
    # NOTE: this uses the weld's LAST frame, i.e. the future. That is exactly why
    # it is only allowed in the "cheating" baseline, never in my real model.
    out = df.copy()
    frames = out.groupby("weld")["frame"]
    first, last = frames.transform("min"), frames.transform("max")
    out["position"] = (out["frame"] - first) / (last - first).replace(0, 1)
    return out


def make_model():
    # Logistic regression: a simple, explainable model that draws a straight
    # dividing line between Sound and Unsound.
    # StandardScaler first rescales every feature to a similar size, which
    # logistic regression needs to work well.
    # class_weight="balanced": there are 2x more Sound frames; this stops the
    # model from simply predicting "Sound" most of the time.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def evaluate(df, columns, n_folds=N_FOLDS):
    # I train and test the model n_folds times, each time on different welds,
    # and collect three scores per round.
    X, y, groups = df[list(columns)].to_numpy(), df["y"].to_numpy(), df["weld"].to_numpy()
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_folds).split(X, y, groups):
        # Safety check: no weld may appear in both training and testing.
        assert not set(groups[train_idx]) & set(groups[test_idx])
        model = make_model().fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]   # probability of "Unsound"
        pred = (prob >= 0.5).astype(int)                # 1 if probability >= 50%
        scores.append({
            # ROC AUC: 0.5 = guessing, 1.0 = perfect ranking of unsound above sound.
            "roc_auc": roc_auc_score(y[test_idx], prob),
            # Balanced accuracy: average of "% of Sound correct" and "% of Unsound correct".
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            # F1 for Unsound: balances missed defects and false alarms.
            "f1_unsound": f1_score(y[test_idx], pred),
        })
    return pd.DataFrame(scores)


def run(df):
    # I compare three models with the same weld-based splits.
    df = add_position(df)
    experiments = {
        "position only (cheating baseline)": ["position"],
        "thermal features (my model)": FEATURE_COLUMNS,
        "thermal features + position": FEATURE_COLUMNS + ["position"],
    }
    rows = []
    for name, columns in experiments.items():
        s = evaluate(df, columns)
        # I report the average over the 5 rounds, and the spread (std) between rounds.
        row = {"model": name}
        for metric in s.columns:
            row[metric] = f"{s[metric].mean():.3f} ± {s[metric].std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# This block runs only when I start the file directly: python -m laser_defect.train
if __name__ == "__main__":
    df = load_features()
    print(f"{len(df)} frames from {df['weld'].nunique()} welds, {N_FOLDS}-fold split by weld\n")
    results = run(df)
    print(results.to_string(index=False))
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {RESULTS_FILE}")
