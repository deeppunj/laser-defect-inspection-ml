# test_explain.py
# Here I check that SHAP finds the feature I KNOW carries the signal.

import matplotlib
matplotlib.use("Agg")      # I draw without opening windows during tests
import numpy as np

from laser_defect.explain import explain, plot_dependence, plot_importance
from laser_defect.model import train_final
from tests_helpers import fake_features


def test_shap_finds_the_real_signal():
    # In my fake data only contrast_rel differs between Sound and Unsound.
    df = fake_features(signal=True)
    bundle = train_final(df, n_folds=3)
    X, sv, imp = explain(bundle, df, n_sample=100)
    assert imp["feature"].iloc[0] == "contrast_rel"
    # LOW contrast means Unsound, so the direction must be negative.
    assert imp.loc[0, "direction"] < 0
    assert sv.shape == X.shape
    assert len(plot_importance(imp).axes) == 1
    assert len(plot_dependence(X, sv, bundle["features"], "contrast_rel").axes) == 1
