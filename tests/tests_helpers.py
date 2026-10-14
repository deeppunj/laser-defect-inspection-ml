# tests_helpers.py
# Shared fake data for my tests, so I don't copy the same code into several files.

import numpy as np
import pandas as pd

from laser_defect.features import FEATURE_COLUMNS


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
