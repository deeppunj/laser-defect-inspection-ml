# test_pipeline.py
# Phase 7: one end-to-end test. Fake thermal frames go through my whole chain
# (features -> Random Forest -> SHAP) exactly as real data would.

import numpy as np
import pandas as pd

from laser_defect.explain import explain
from laser_defect.features import FEATURE_COLUMNS, build_features
from laser_defect.model import evaluate, train_final


def test_frames_to_explanation():
    rng = np.random.default_rng(0)
    frames, rows = [], []
    for w in range(6):
        for f in range(40):
            y = int(f >= 25)                        # last 15 frames unsound
            img = np.full((32, 32), 430.0) + rng.normal(0, 0.3, (32, 32))
            img[14:18, :] += 20 if y == 0 else 8    # unsound = weaker hot band
            frames.append(img.astype(np.float32))
            rows.append({"weld": f"001-{w:03d}", "frame": f, "y": y})
    df = build_features(np.stack(frames), pd.DataFrame(rows))
    scores, _ = evaluate(df, n_folds=3, inner_folds=2)
    assert scores["roc_auc"].mean() > 0.9
    _, _, imp = explain(train_final(df, n_folds=3), df, n_sample=100)
    assert set(imp["feature"]) == set(FEATURE_COLUMNS)
