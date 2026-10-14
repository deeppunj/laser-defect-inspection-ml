# dashboard.py
# Phase 8: the calculations behind my Streamlit dashboard (app.py).
# I keep them here, not in app.py, so I can test them with pytest.

import pandas as pd


def predict_weld(bundle, df, weld):
    # I take all frames of ONE weld, in time order, and add the model's
    # probability of "Unsound" and its decision (using my saved threshold).
    w = df[df["weld"] == weld].sort_values("frame").copy()
    w["prob_unsound"] = bundle["model"].predict_proba(w[bundle["features"]].to_numpy())[:, 1]
    w["pred"] = (w["prob_unsound"] >= bundle["threshold"]).astype(int)
    return w


def weld_summary(w):
    # Simple numbers for one weld: how many frames, how many truly unsound,
    # how many the model flagged, and how often it agreed with the label.
    return {"frames": len(w),
            "true_unsound": int(w["y"].sum()),
            "flagged_unsound": int(w["pred"].sum()),
            "agreement": float((w["pred"] == w["y"]).mean())}


def first_alarm(w):
    # The first frame the model flags as Unsound, and the first truly unsound frame.
    # Comparing them tells me if the alarm comes early, on time, or late.
    pred = w.loc[w["pred"] == 1, "frame"]
    true = w.loc[w["y"] == 1, "frame"]
    return (int(pred.iloc[0]) if len(pred) else None,
            int(true.iloc[0]) if len(true) else None)
