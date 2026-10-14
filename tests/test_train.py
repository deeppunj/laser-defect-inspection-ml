# test_train.py
# Here I test the training code on a small fake dataset where I know the truth.

import numpy as np
import pandas as pd

from laser_defect.features import FEATURE_COLUMNS
from laser_defect.train import add_position, evaluate


def fake_features(signal=True, n_welds=10, n_frames=20, seed=0):
    # Each weld: first half Sound, second half Unsound.
    # If signal=True, contrast_rel is lower for Unsound; otherwise all features are noise.
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_welds):
        for f in range(n_frames):
            y = int(f >= n_frames // 2)
            row = {c: rng.normal() for c in FEATURE_COLUMNS}
            if signal:
                row["contrast_rel"] = rng.normal(1.0 - 0.8 * y, 0.1)
            rows.append({"weld": f"001-{w:03d}", "frame": f, "y": y, **row})
    return pd.DataFrame(rows)


def test_position_goes_from_0_to_1():
    df = add_position(fake_features())
    assert df["position"].min() == 0 and df["position"].max() == 1


def test_real_signal_is_learned():
    scores = evaluate(fake_features(signal=True), FEATURE_COLUMNS)
    assert scores["roc_auc"].mean() > 0.95


def test_pure_noise_gives_chance_level():
    # With no signal, an honest weld-based evaluation must give about 0.5.
    scores = evaluate(fake_features(signal=False), FEATURE_COLUMNS)
    assert 0.35 < scores["roc_auc"].mean() < 0.65
