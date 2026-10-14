# test_data.py
# Here I test data.py on a TINY fake dataset that I build myself inside a
# temporary folder. That way the tests run in a second, without the real 90 MB data.

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from laser_defect.data import build_dataset, load_dataset, load_labels, load_weld


def make_fake_lwir(tmp_path, weld="001-001", labels=None, skip_image=None, bad_shape=False, base=400.0):
    # I build the same folder layout as the real dataset, with a few frames.
    if labels is None:
        labels = ["Beginning", "Sound", "Sound", "Unsound"]
    lwir = tmp_path / "LWIR"
    (lwir / "Labels").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Frame": range(len(labels)), "Label": labels}).to_csv(
        lwir / "Labels" / f"{weld}.csv")

    for frame, label in enumerate(labels):
        if label == "Beginning" or frame == skip_image:
            continue                              # like the real data: no image for Beginning
        folder = lwir / "Labeled_Frames" / weld / label.lower()
        folder.mkdir(parents=True, exist_ok=True)
        shape = (16, 16) if bad_shape else (32, 32)
        # float32 pixels give a 32-bit decimal ("F") image, the same type as the real thermal frames
        pixels = np.full(shape, base + frame, dtype=np.float32)
        Image.fromarray(pixels).save(folder / f"{weld}-{frame:03d}.tiff")
    return lwir


def test_load_weld_skips_beginning_and_sets_labels(tmp_path):
    lwir = make_fake_lwir(tmp_path)
    images, rows, missing = load_weld("001-001", lwir)
    assert len(images) == 3                       # 2 Sound + 1 Unsound; Beginning left out
    assert [r["y"] for r in rows] == [0, 0, 1]    # Unsound = 1
    assert rows[0]["speed_m_per_min"] == 1        # "001" means 1 m/min
    assert missing == []


def test_missing_image_is_reported_not_invented(tmp_path):
    lwir = make_fake_lwir(tmp_path, skip_image=2)
    images, rows, missing = load_weld("001-001", lwir)
    assert missing == [2]
    assert len(images) == len(rows) == 2


def test_unknown_label_is_rejected(tmp_path):
    lwir = make_fake_lwir(tmp_path, labels=["Sound", "Broken"])
    with pytest.raises(ValueError):
        load_labels("001-001", lwir)


def test_wrong_image_size_is_rejected(tmp_path):
    lwir = make_fake_lwir(tmp_path, bad_shape=True)
    with pytest.raises(ValueError):
        load_weld("001-001", lwir)


def test_build_and_reload_give_the_same_data(tmp_path):
    lwir = make_fake_lwir(tmp_path)
    make_fake_lwir(tmp_path, weld="002-005")      # a second weld at 5 m/min
    out = tmp_path / "frames.npz"

    X, table, problems = build_dataset(lwir, out)
    X2, table2 = load_dataset(out)

    assert X.shape == (6, 32, 32)
    assert np.array_equal(X, X2)                  # the saved images are identical
    assert list(table2["weld"].unique()) == ["001-001", "002-005"]
    assert set(table2["speed_m_per_min"]) == {1, 5}


def test_high_camera_scale_is_flagged_per_weld(tmp_path):
    # One weld on the normal scale (~400), one on the high scale (~36,000).
    lwir = make_fake_lwir(tmp_path, weld="001-001", base=400.0)
    make_fake_lwir(tmp_path, weld="001-020", base=36000.0)
    X, table, problems = build_dataset(lwir, tmp_path / "frames.npz")
    flags = table.groupby("weld")["high_scale"].first()
    assert not flags["001-001"]
    assert flags["001-020"]
