# explore.py
# Phase 2: here I keep the functions for exploring the data and drawing plots.
# I put them in a .py file (not only in the notebook) so that:
#   1. I can test them with pytest, and
#   2. I can re-create every figure with one command:
#          python -m laser_defect.explore
# The notebook notebooks/01_exploratory_analysis.ipynb calls these functions
# and explains what each plot shows.

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laser_defect.data import load_dataset

# Where I save the figures that the README shows.
FIG_DIR = Path("reports/figures")

# I use the same two colours everywhere, so Sound/Unsound are easy to recognise.
COLOURS = {"Sound": "tab:blue", "Unsound": "tab:red"}


def frame_stats(X, table):
    # I describe every 32 x 32 frame with a few simple numbers.
    # X has shape (frames, 32, 32); I flatten each frame to 1,024 pixel values.
    flat = X.reshape(len(X), -1)
    stats = table.copy()
    stats["mean"] = flat.mean(axis=1)                    # average brightness (overall heat)
    stats["max"] = flat.max(axis=1)                      # hottest pixel (the melt pool)
    stats["std"] = flat.std(axis=1)                      # spread: a sharp hot spot gives a large spread
    stats["range"] = stats["max"] - flat.min(axis=1)     # hottest minus coldest pixel
    return stats


def add_weld_relative(stats, columns=("mean", "max", "std", "range")):
    # The 29 high-scale welds have values ~80x larger, so I cannot compare raw numbers
    # across welds. For EXPLORING, I express every value relative to its own weld:
    #     z = (value - weld average) / weld standard deviation
    # z = 0 means "typical for this weld", z = -1 means "one spread lower than usual".
    # This uses no labels, so it does not leak the answer.
    # (Phase 3 decides the normalisation used for modelling, more carefully.)
    out = stats.copy()
    for column in columns:
        group = out.groupby("weld")[column]
        out[f"{column}_z"] = (out[column] - group.transform("mean")) / group.transform("std")
    return out


def add_position(stats):
    # I compute where each frame sits inside its weld: 0 = first labelled frame,
    # 1 = last. I need this to check whether "Unsound" simply means "late in the weld".
    out = stats.copy()
    frames = out.groupby("weld")["frame"]
    first, last = frames.transform("min"), frames.transform("max")
    out["position"] = (out["frame"] - first) / (last - first).replace(0, 1)
    return out


def plot_example_frames(X, stats, weld, n=3):
    # I show n Sound and n Unsound frames from ONE weld, evenly spaced in time.
    # I use one shared colour scale per row so brightness can be compared.
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for row, label in enumerate(("Sound", "Unsound")):
        idx = stats.index[(stats["weld"] == weld) & (stats["label"] == label)]
        if len(idx) == 0:
            raise ValueError(f"weld {weld} has no {label} frames")
        picks = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
        vmin, vmax = X[picks].min(), X[picks].max()
        for col, i in enumerate(picks):
            ax = axes[row, col]
            im = ax.imshow(X[i], cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} – frame {stats.loc[i, 'frame']}", color=COLOURS[label])
            ax.axis("off")
        fig.colorbar(im, ax=axes[row].tolist(), shrink=0.8, label="camera value")
    fig.suptitle(f"Thermal frames from weld {weld}")
    return fig


def plot_weld_timeline(stats, welds):
    # For a few welds, I plot the hottest pixel over time and colour each point
    # by its label. This shows WHEN a weld turns unsound and what changes.
    fig, axes = plt.subplots(len(welds), 1, figsize=(9, 2.6 * len(welds)), squeeze=False)
    for ax, weld in zip(axes[:, 0], welds):
        w = stats[stats["weld"] == weld]
        for label, colour in COLOURS.items():
            part = w[w["label"] == label]
            ax.scatter(part["frame"], part["max"], s=6, color=colour, label=label)
        ax.set_title(f"Weld {weld} ({w['speed_m_per_min'].iloc[0]} m/min)")
        ax.set_ylabel("hottest pixel")
    axes[-1, 0].set_xlabel("frame number (30 frames per second)")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_camera_scales(stats):
    # I plot every weld's typical (median) pixel value. Two separate groups
    # appear: the two camera scales I found in phase 1.
    per_weld = stats.groupby("weld").agg(median=("mean", "median"), high=("high_scale", "first"))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    colours = np.where(per_weld["high"], "tab:orange", "tab:green")
    ax.scatter(range(len(per_weld)), per_weld["median"], c=colours, s=18)
    ax.set_yscale("log")                     # log scale, because the groups differ ~80x
    ax.set_xlabel("weld (sorted by name)")
    ax.set_ylabel("median pixel value (log)")
    ax.set_title("Camera scales: orange = high scale (29 welds), green = standard (68 welds)")
    fig.tight_layout()
    return fig


def plot_feature_by_label(stats, columns=("mean_z", "max_z", "std_z", "range_z")):
    # I compare the weld-relative features between Sound and Unsound with box plots.
    # If the boxes barely overlap, that feature separates the classes well.
    fig, axes = plt.subplots(1, len(columns), figsize=(3.2 * len(columns), 3.6))
    for ax, column in zip(axes, columns):
        data = [stats.loc[stats["label"] == label, column] for label in COLOURS]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        ax.set_xticks([1, 2], list(COLOURS))   # I name the two boxes Sound and Unsound
        for patch, colour in zip(box["boxes"], COLOURS.values()):
            patch.set_facecolor(colour)
            patch.set_alpha(0.5)
        ax.set_title(column.replace("_z", " (weld-relative)"))
        ax.axhline(0, color="grey", lw=0.5)
    fig.tight_layout()
    return fig


def plot_position_by_label(stats):
    # The warning plot: where in the weld do Sound and Unsound frames occur?
    # If Unsound frames are almost always late, a model could cheat by learning
    # "late = unsound". I test this properly in phase 4.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, 1, 21)
    for label, colour in COLOURS.items():
        ax.hist(stats.loc[stats["label"] == label, "position"], bins=bins,
                alpha=0.6, color=colour, label=label)
    ax.set_xlabel("position inside the weld (0 = start, 1 = end)")
    ax.set_ylabel("number of frames")
    ax.set_title("Unsound frames come late in the weld: a leakage risk")
    ax.legend()
    fig.tight_layout()
    return fig


def prepare(path=None):
    # I load the phase 1 data and add every column the plots need, in one call.
    X, table = load_dataset() if path is None else load_dataset(path)
    stats = add_position(add_weld_relative(frame_stats(X, table)))
    return X, stats


def save_all_figures(fig_dir=FIG_DIR, example_weld="001-020",
                     timeline_welds=("001-020", "001-040", "002-030")):
    # I create every figure and save it as a .png for the README.
    X, stats = prepare()
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "example_frames.png": plot_example_frames(X, stats, example_weld),
        "weld_timeline.png": plot_weld_timeline(stats, list(timeline_welds)),
        "camera_scales.png": plot_camera_scales(stats),
        "features_by_label.png": plot_feature_by_label(stats),
        "position_by_label.png": plot_position_by_label(stats),
    }
    for name, fig in figures.items():
        fig.savefig(fig_dir / name, dpi=120, bbox_inches="tight")
        plt.close(fig)                       # I close each figure to free memory
        print(f"Saved {fig_dir / name}")
    return stats


# This block runs only when I start the file directly: python -m laser_defect.explore
if __name__ == "__main__":
    matplotlib.use("Agg")                    # I draw to files only, no windows pop up
    stats = save_all_figures()
    print()
    print("Median weld-relative values per label:")
    print(stats.groupby("label")[["mean_z", "max_z", "std_z", "position"]].median().round(2))
