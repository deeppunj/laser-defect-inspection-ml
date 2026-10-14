# features.py
# Phase 3: here I turn every 32 x 32 thermal frame into a few numbers ("features")
# that describe the physics of the weld, so a model can learn from them.
#
# I run it from the project folder with:
#     python -m laser_defect.features
# It saves data/processed/features.csv (one row per frame).
#
# Two rules I follow so the features are honest:
#   1. NO FUTURE INFORMATION. A feature of frame t may only use frame t, earlier
#      frames, and the start of the same weld. In a real welding line I would
#      not know the future either.
#   2. NO LABELS. Features never look at Sound/Unsound.

from pathlib import Path

import numpy as np
import pandas as pd

from laser_defect.data import load_dataset

OUT_FILE = Path("data/processed/features.csv")

# How many of the first labelled frames of each weld I use as its "baseline"
# (15 frames = 0.5 s at 30 frames per second).
BASELINE_FRAMES = 15

# The features a model may use. I keep this list in one place, so phases 4-8
# all use exactly the same columns.
FEATURE_COLUMNS = [
    "contrast_rel",     # hot spot strength, relative to the weld's start
    "mean_rel",         # overall heat, relative to the weld's start
    "hot_area",         # fraction of pixels that are "hot" (melt pool / hot band size)
    "band_width",       # width of the hot band across the seam (pixels)
    "band_row",         # where the hot band sits vertically (pixels)
    "asymmetry",        # left-right heat difference along the seam
    "contrast_change",  # change in contrast since the previous frame
]


def frame_features(frame):
    # I compute shape features of ONE frame. "Shape" features do not depend on the
    # camera scale, because I first rescale the frame to 0 (coldest) ... 1 (hottest).
    lo, hi = frame.min(), frame.max()
    contrast = hi - np.median(frame)            # how much the hot spot stands out
    span = hi - lo
    if span == 0:                                # a perfectly flat frame: no hot spot at all
        return {"contrast": 0.0, "mean": float(frame.mean()),
                "hot_area": 0.0, "band_width": 0.0, "band_row": 15.5, "asymmetry": 0.0}
    scaled = (frame - lo) / span                 # now every pixel is between 0 and 1

    # Hot area: share of pixels above half of the maximum (a "half-maximum" area,
    # like measuring a laser spot size at half its peak).
    hot_area = float((scaled > 0.5).mean())

    # The weld seam runs left-right in the image, so I average each ROW to get a
    # vertical heat profile (32 numbers, one per row).
    profile = scaled.mean(axis=1)
    profile = profile - profile.min()            # I remove the background level
    # Band width: how many rows are above half of the profile's peak (FWHM in pixels).
    band_width = float((profile > 0.5 * profile.max()).sum()) if profile.max() > 0 else 0.0
    # Band row: the "centre of mass" of the profile, i.e. where the hot band is.
    rows = np.arange(len(profile))
    band_row = float((rows * profile).sum() / profile.sum()) if profile.sum() > 0 else 15.5

    # Asymmetry: left half minus right half, on the 0..1 scale.
    # It tells me whether the heat is spread evenly along the seam.
    half = scaled.shape[1] // 2
    asymmetry = float(scaled[:, :half].mean() - scaled[:, half:].mean())

    return {"contrast": float(contrast), "mean": float(frame.mean()),
            "hot_area": hot_area, "band_width": band_width,
            "band_row": band_row, "asymmetry": asymmetry}


def add_weld_relative(df, baseline_frames=BASELINE_FRAMES):
    # The camera scale differs ~80x between welds, so raw "contrast" and "mean"
    # are not comparable. I compare each frame with the START of its own weld:
    #     relative value = (value - baseline value) / baseline contrast
    # The baseline uses only the first frames of the weld (rule 1: no future).
    out = df.sort_values(["weld", "frame"]).copy()
    start = out.groupby("weld").head(baseline_frames)          # the first frames of every weld
    base = start.groupby("weld")[["contrast", "mean"]].median()
    base_contrast = out["weld"].map(base["contrast"]).replace(0, np.nan)
    base_mean = out["weld"].map(base["mean"])

    out["contrast_rel"] = out["contrast"] / base_contrast      # 1.0 = as strong as at the start
    out["mean_rel"] = (out["mean"] - base_mean) / base_contrast

    # Change since the previous frame of the SAME weld (only looks backwards).
    # The first frame of each weld has no previous frame, so I set its change to 0.
    out["contrast_change"] = out.groupby("weld")["contrast_rel"].diff().fillna(0.0)
    return out


def build_features(X, table):
    # I compute the features for every frame and join them to the frame table.
    rows = [frame_features(frame) for frame in X]
    df = pd.concat([table.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df = add_weld_relative(df)
    # Final check: no missing or infinite values in the model features.
    bad = ~np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all(axis=1)
    if bad.any():
        raise ValueError(f"{bad.sum()} frames have invalid feature values")
    return df


def load_features(path=OUT_FILE):
    # Later phases call this to get the feature table.
    return pd.read_csv(path)


# This block runs only when I start the file directly: python -m laser_defect.features
if __name__ == "__main__":
    X, table = load_dataset()
    print(f"Computing features for {len(X)} frames...")
    df = build_features(X, table)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} ({len(df)} rows, {len(FEATURE_COLUMNS)} features)\n")
    print("Median of each feature per label:")
    print(df.groupby("label")[FEATURE_COLUMNS].median().round(3).T)
