# test_features.py
# Here I test my features on small hand-made frames where I KNOW the right answer.

import numpy as np
import pandas as pd
import pytest

from laser_defect.features import FEATURE_COLUMNS, add_weld_relative, build_features, frame_features


def band_frame(width=4, centre=16, base=430.0, peak=20.0):
    # I draw a horizontal hot band: rows centre-width/2 ... centre+width/2 are hot.
    f = np.full((32, 32), base, dtype=np.float32)
    f[centre - width // 2: centre + width // 2, :] += peak
    return f


def test_band_width_and_row_are_measured():
    feats = frame_features(band_frame(width=4, centre=16))
    assert feats["band_width"] == 4               # 4 hot rows
    assert feats["band_row"] == pytest.approx(15.5)  # rows 14-17 -> centre 15.5
    assert feats["hot_area"] == pytest.approx(4 / 32)


def test_asymmetry_sign():
    f = band_frame()
    f[:, :16] += 5                                 # left half hotter
    assert frame_features(f)["asymmetry"] > 0


def test_flat_frame_has_no_hot_spot():
    feats = frame_features(np.full((32, 32), 440.0, dtype=np.float32))
    assert feats["contrast"] == 0 and feats["hot_area"] == 0


def test_shape_features_ignore_camera_scale():
    # The same frame x80 (the "high camera scale") must give the same shape features.
    a, b = frame_features(band_frame()), frame_features(band_frame() * 80)
    for key in ("hot_area", "band_width", "band_row", "asymmetry"):
        assert a[key] == pytest.approx(b[key])


def make_weld(weld, peaks, scale=1.0):
    # One weld = a list of frames whose hot band gets weaker over time.
    X = np.stack([band_frame(peak=p) * scale for p in peaks])
    table = pd.DataFrame({"weld": weld, "frame": range(len(peaks)), "y": 0, "label": "Sound"})
    return X, table


def test_relative_features_remove_camera_scale():
    peaks = [20, 20, 20, 10, 5]
    X1, t1 = make_weld("001-020", peaks)
    X2, t2 = make_weld("001-001", peaks, scale=80)   # same weld, 80x camera scale
    df = build_features(np.concatenate([X1, X2]), pd.concat([t1, t2]))
    w1 = df[df.weld == "001-020"]["contrast_rel"].to_numpy()
    w2 = df[df.weld == "001-001"]["contrast_rel"].to_numpy()
    assert np.allclose(w1, w2)
    assert w1[-1] < w1[0]                          # weaker hot band -> lower relative contrast


def test_baseline_does_not_use_the_future():
    # If I change only LATE frames, the features of EARLY frames must not change.
    X, t = make_weld("001-020", [20] * 20 + [10] * 5)
    X_changed = X.copy()
    X_changed[-5:] *= 0.5                           # change only the last 5 frames
    a = build_features(X, t).iloc[:20]
    b = build_features(X_changed, t).iloc[:20]
    assert np.allclose(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])


def test_contrast_change_starts_at_zero_per_weld():
    df = pd.DataFrame({"weld": ["a", "a", "b", "b"], "frame": [0, 1, 0, 1],
                       "contrast": [1.0, 2.0, 1.0, 1.0], "mean": [0.0] * 4})
    out = add_weld_relative(df, baseline_frames=1)
    assert out.groupby("weld")["contrast_change"].first().tolist() == [0.0, 0.0]
