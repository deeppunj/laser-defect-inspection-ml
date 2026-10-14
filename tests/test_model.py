# test_model.py
# Here I test the Random Forest code on small fake data where I know the truth.

import numpy as np

from laser_defect.model import best_threshold, evaluate, train_final
from tests_helpers import fake_features


def test_threshold_is_chosen_from_the_grid():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.3, 0.35, 0.9])
    # 0.35 separates perfectly; my grid contains 0.35.
    assert best_threshold(y, prob) == 0.35


def test_forest_learns_a_nonlinear_rule():
    # Unsound only when TWO features are both low: a straight line can't do this well.
    df = fake_features(signal=False, n_welds=10, n_frames=30)
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=len(df)), rng.normal(size=len(df))
    df["contrast_rel"], df["mean_rel"] = a, b
    df["y"] = ((a < 0) & (b < 0)).astype(int)
    scores, cm = evaluate(df, n_folds=3, inner_folds=2)
    assert scores["roc_auc"].mean() > 0.9
    assert cm.sum() == len(df)                 # every frame tested exactly once


def test_final_bundle_contains_model_threshold_features():
    bundle = train_final(fake_features(signal=True), n_folds=3)
    assert set(bundle) == {"model", "threshold", "features"}
    assert 0.2 <= bundle["threshold"] <= 0.8
