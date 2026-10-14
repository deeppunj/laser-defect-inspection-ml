# data.py
# Phase 1: here I load the thermal (LWIR) camera frames and their labels,
# check that the data is consistent, and save everything into ONE file
# that later phases can load in under a second.
#
# I run it from the project folder with:
#     python -m laser_defect.data
#
# What the raw data looks like (I checked this before writing the code):
#   data/raw/Dataset/Preprocessed/LWIR/
#     Labels/001-020.csv                 -> one table per weld: Frame, Label
#     Labeled_Frames/001-020/sound/      -> 32 x 32 thermal images labelled Sound
#     Labeled_Frames/001-020/unsound/    -> 32 x 32 thermal images labelled Unsound
#   Weld names look like "001-020": the first part is the welding speed group
#   (001 = 1 m/min, 002 = 5 m/min), the second part is the weld number.
#   "Beginning" frames (before the weld really starts) have no image in
#   Labeled_Frames, so I leave them out.

from pathlib import Path

import numpy as np                # I use numpy arrays to hold the images (grids of numbers)
import pandas as pd               # I use pandas for the label tables
from PIL import Image             # I use Pillow to open .tiff image files

# The folder my download script created in phase 0.
LWIR_DIR = Path("data/raw/Dataset/Preprocessed/LWIR")

# Where I save the cleaned result of this phase.
OUT_FILE = Path("data/processed/frames.npz")

# The only labels I accept. Anything else means the data is not what I expect.
VALID_LABELS = {"Sound", "Unsound", "Beginning"}

# I translate the speed group in the weld name into the real welding speed.
SPEED_M_PER_MIN = {"001": 1, "002": 5}

# Every thermal frame must be a 32 x 32 grid of pixels.
FRAME_SHAPE = (32, 32)

# While checking the data I found that 29 welds (the earliest recordings, weld
# numbers 000-014) store pixel values around 11,000-40,000, while all other welds
# are around 420-460. No weld mixes both. I mark each weld's scale, so later I can
# normalise per weld instead of mixing two different units in one model.
HIGH_SCALE_THRESHOLD = 1000.0


def list_welds(lwir_dir=LWIR_DIR):
    # I find every label file and use its file name (without ".csv") as the weld id.
    files = sorted((lwir_dir / "Labels").glob("*.csv"))
    if not files:
        # I stop with a clear message if the data is not there, instead of
        # quietly returning nothing and getting a confusing error later.
        raise FileNotFoundError(
            f"No label files found in {lwir_dir / 'Labels'}. "
            "Did you run: python -m laser_defect.download ?"
        )
    return [f.stem for f in files]        # .stem = file name without extension


def load_labels(weld, lwir_dir=LWIR_DIR):
    # I read the label table of one weld. It has the columns: (index), Frame, Label.
    table = pd.read_csv(lwir_dir / "Labels" / f"{weld}.csv")

    # Check 1: the columns I need must exist.
    for column in ("Frame", "Label"):
        if column not in table.columns:
            raise ValueError(f"{weld}: label file has no '{column}' column")

    # Check 2: only known labels.
    unknown = set(table["Label"]) - VALID_LABELS
    if unknown:
        raise ValueError(f"{weld}: unknown labels {unknown}")

    # Check 3: every frame number appears only once.
    if table["Frame"].duplicated().any():
        raise ValueError(f"{weld}: duplicate frame numbers in label file")

    # I keep only the two columns I need, sorted by time.
    return table[["Frame", "Label"]].sort_values("Frame").reset_index(drop=True)


def load_frame(path):
    # I open one .tiff image and turn it into a 32 x 32 numpy array of decimals.
    with Image.open(path) as image:
        frame = np.array(image, dtype=np.float32)

    # Check 4: the image has the right size.
    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"{path.name}: expected shape {FRAME_SHAPE}, got {frame.shape}")

    # Check 5: no broken pixels (NaN = "not a number", inf = infinity).
    if not np.isfinite(frame).all():
        raise ValueError(f"{path.name}: contains NaN or infinite pixel values")

    return frame


def load_weld(weld, lwir_dir=LWIR_DIR):
    # I load all Sound and Unsound frames of one weld, in time order.
    # I return: the images, a table describing each image, and a list of
    # frames that were labelled but had no image file.
    labels = load_labels(weld, lwir_dir)
    labels = labels[labels["Label"] != "Beginning"]   # Beginning frames have no images

    images, rows, missing = [], [], []
    for frame_number, label in zip(labels["Frame"], labels["Label"]):
        # The image file name is built from weld id + 3-digit frame number,
        # e.g. frame 70 of weld 001-020 -> sound/001-020-070.tiff
        path = lwir_dir / "Labeled_Frames" / weld / label.lower() / f"{weld}-{frame_number:03d}.tiff"
        if not path.exists():
            missing.append(int(frame_number))     # I note it and move on
            continue
        images.append(load_frame(path))
        rows.append({
            "weld": weld,
            "speed_m_per_min": SPEED_M_PER_MIN[weld[:3]],
            "frame": int(frame_number),
            "label": label,
            # I store the label also as a number, because models need numbers:
            # 1 = Unsound (the defect I want to detect), 0 = Sound.
            "y": int(label == "Unsound"),
        })

    return images, rows, missing


def build_dataset(lwir_dir=LWIR_DIR, out_file=OUT_FILE):
    # I loop over all welds, collect every frame, and save one compact file.
    all_images, all_rows, problems = [], [], {}

    welds = list_welds(lwir_dir)
    for i, weld in enumerate(welds, start=1):
        images, rows, missing = load_weld(weld, lwir_dir)
        all_images.extend(images)
        all_rows.extend(rows)
        if missing:
            problems[weld] = missing
        # I print progress every 10 welds so I can see the script is working.
        if i % 10 == 0 or i == len(welds):
            print(f"Loaded {i}/{len(welds)} welds")

    # I stack the list of 32 x 32 images into one 3-D array: (number of frames, 32, 32).
    X = np.stack(all_images)
    table = pd.DataFrame(all_rows)

    # I compute the typical (median) pixel value of every frame, then decide
    # per weld whether it is on the high camera scale.
    frame_median = np.median(X.reshape(len(X), -1), axis=1)
    weld_median = pd.Series(frame_median).groupby(table["weld"]).transform("median")
    table["high_scale"] = (weld_median > HIGH_SCALE_THRESHOLD).to_numpy()

    # Final check: one table row per image, otherwise everything later is misaligned.
    assert len(table) == len(X), "number of images and table rows differ"

    # I save the images and the table columns together in one compressed file.
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        X=X,
        # I save weld names as plain text (not Python objects), so they reload safely.
        weld=table["weld"].to_numpy().astype(str),
        speed_m_per_min=table["speed_m_per_min"].to_numpy(),
        frame=table["frame"].to_numpy(),
        y=table["y"].to_numpy(),
        high_scale=table["high_scale"].to_numpy(),
    )
    return X, table, problems


def load_dataset(path=OUT_FILE):
    # Later phases call this to get the data back quickly, without re-reading 16,000 files.
    with np.load(path, allow_pickle=False) as d:
        X = d["X"]
        table = pd.DataFrame({
            "weld": d["weld"],
            "speed_m_per_min": d["speed_m_per_min"],
            "frame": d["frame"],
            "y": d["y"],
            "high_scale": d["high_scale"],
        })
    table["label"] = np.where(table["y"] == 1, "Unsound", "Sound")
    return X, table


def print_summary(X, table, problems):
    # I print a short report, so I can check the result with my own eyes.
    print()
    print(f"Frames: {len(X)}   image shape: {X.shape[1:]}   welds: {table['weld'].nunique()}")
    print(f"Pixel values: min {X.min():.1f}, max {X.max():.1f} (camera units)")
    print()
    print("Frames per label and welding speed:")
    print(pd.crosstab(table["label"], table["speed_m_per_min"], margins=True,
                      colnames=["speed (m/min)"]))
    print()
    high = sorted(table.loc[table["high_scale"], "weld"].unique())
    if high:
        print(f"Note: {len(high)} welds use a high camera scale (~11,000-40,000 instead of ~430): "
              f"{high[0]} ... {high[-1]}. I will normalise per weld in phase 3.")
    if problems:
        for weld, frames in problems.items():
            print(f"Warning: {weld} has {len(frames)} labelled frames without an image "
                  f"(frames {frames[0]}-{frames[-1]}); I left them out.")
    else:
        print("No missing images.")


# This block runs only when I start the file directly: python -m laser_defect.data
if __name__ == "__main__":
    X, table, problems = build_dataset()
    print_summary(X, table, problems)
    print(f"\nSaved {OUT_FILE}")
