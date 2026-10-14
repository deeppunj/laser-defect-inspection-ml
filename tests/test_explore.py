# test_explore.py
# Here I test my exploration functions on a tiny fake dataset, and (only if the
# real data exists on this computer) I run the whole notebook from top to bottom.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # I draw without opening windows during tests
import numpy as np
import pandas as pd
import pytest

from laser_defect.explore import (
    add_position, add_weld_relative, frame_stats, plot_example_frames,
    plot_feature_by_label, plot_position_by_label, plot_weld_timeline,
)


def fake_data():
    # Two welds x 4 frames. Sound frames are hot (450), Unsound frames cooler (440).
    labels = ["Sound", "Sound", "Unsound", "Unsound"] * 2
    X = np.stack([np.full((32, 32), 450.0 if l == "Sound" else 440.0, dtype=np.float32)
                  for l in labels])
    X[:, 16, 16] += 5                    # one hot pixel, so std is not zero
    X[4:] *= 80                          # weld 2 on the "high camera scale"
    table = pd.DataFrame({
        "weld": ["001-001"] * 4 + ["001-020"] * 4,
        "speed_m_per_min": 1,
        "frame": [10, 11, 12, 13] * 2,
        "y": [int(l == "Unsound") for l in labels],
        "label": labels,
        "high_scale": [False] * 4 + [True] * 4,
    })
    return X, table


def test_frame_stats_values():
    X, table = fake_data()
    stats = frame_stats(X, table)
    assert stats.loc[0, "max"] == pytest.approx(455.0)
    assert stats.loc[0, "range"] == pytest.approx(5.0)


def test_weld_relative_removes_camera_scale():
    # After z-scoring, both welds must give the SAME numbers, even though
    # weld 2 is 80x larger. That is exactly why I normalise per weld.
    X, table = fake_data()
    stats = add_weld_relative(frame_stats(X, table))
    weld1 = stats.loc[stats["weld"] == "001-001", "mean_z"].to_numpy()
    weld2 = stats.loc[stats["weld"] == "001-020", "mean_z"].to_numpy()
    assert np.allclose(weld1, weld2)


def test_position_runs_from_0_to_1():
    X, table = fake_data()
    stats = add_position(frame_stats(X, table))
    assert stats["position"].min() == 0.0
    assert stats["position"].max() == 1.0


def test_plots_are_created():
    X, table = fake_data()
    stats = add_position(add_weld_relative(frame_stats(X, table)))
    for fig in (plot_example_frames(X, stats, "001-001", n=2),
                plot_weld_timeline(stats, ["001-001"]),
                plot_feature_by_label(stats),
                plot_position_by_label(stats)):
        assert len(fig.axes) > 0


@pytest.mark.skipif(not Path("data/processed/frames.npz").exists(),
                    reason="real data not built yet (run python -m laser_defect.data)")
def test_notebook_runs_without_errors():
    # I execute every cell of the notebook. If any cell raises an error, this test fails.
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read("notebooks/01_exploratory_analysis.ipynb", as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": "notebooks"}}).execute()
