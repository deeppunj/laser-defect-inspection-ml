# explain.py
# Phase 6: here I explain WHY my Random Forest calls a frame Unsound, with SHAP.
#
# I run it from the project folder (after python -m laser_defect.model) with:
#     python -m laser_defect.explain
#
# What is SHAP? For ONE frame, SHAP splits the model's prediction into a
# contribution per feature: "low contrast pushed this frame +0.30 towards
# Unsound, band position pushed it -0.05". Averaging the size of these
# contributions over many frames tells me which features matter most overall.

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from laser_defect.features import load_features
from laser_defect.model import MODEL_FILE

FIG_DIR = Path("reports/figures")
RESULTS_FILE = Path("reports/phase6_shap_importance.csv")
N_SAMPLE = 2000      # I explain 2,000 random frames: enough for stable averages, and fast


def shap_values_unsound(model, X):
    # I compute SHAP values for the "Unsound" class (class 1).
    # Different shap versions return different shapes, so I handle both:
    #   a list [class0, class1]  or  one array (frames, features, classes).
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def importance_table(shap_vals, features):
    # Mean |SHAP| = on average, how much a feature moves the prediction.
    # Mean SHAP sign tells me nothing alone, so I also store the correlation between
    # the feature value and its SHAP value: negative = "LOW values push to Unsound".
    imp = pd.DataFrame({"feature": features,
                        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def direction(X, shap_vals, features):
    # For each feature: correlation between its value and its SHAP value.
    out = {}
    for i, f in enumerate(features):
        x, s = X[:, i], shap_vals[:, i]
        out[f] = float(np.corrcoef(x, s)[0, 1]) if x.std() > 0 and s.std() > 0 else 0.0
    return out


def plot_importance(imp):
    # Bar chart: most important feature at the top.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="tab:orange")
    ax.set_xlabel("mean |SHAP value| (effect on probability of Unsound)")
    ax.set_title("Which thermal features drive the defect prediction")
    fig.tight_layout()
    return fig


def plot_dependence(X, shap_vals, features, name):
    # Scatter for one feature: its value (x) against its push towards Unsound (y).
    i = features.index(name)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, i], shap_vals[:, i], s=4, alpha=0.4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(name)
    ax.set_ylabel("SHAP value (push towards Unsound)")
    ax.set_title(f"How {name} changes the prediction")
    fig.tight_layout()
    return fig


def explain(bundle, df, n_sample=N_SAMPLE, seed=0):
    # I pick random frames, compute SHAP values and return everything I need.
    features = bundle["features"]
    sample = df.sample(min(n_sample, len(df)), random_state=seed)
    X = sample[features].to_numpy()
    sv = shap_values_unsound(bundle["model"], X)
    imp = importance_table(sv, features)
    imp["direction"] = imp["feature"].map(direction(X, sv, features)).round(2)
    return X, sv, imp


# This block runs only when I start the file directly: python -m laser_defect.explain
if __name__ == "__main__":
    bundle = joblib.load(MODEL_FILE)      # the model I saved in phase 5
    X, sv, imp = explain(bundle, load_features())
    print(imp.round(3).to_string(index=False))
    print("\ndirection < 0: LOW values push towards Unsound; > 0: HIGH values do.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_importance(imp).savefig(FIG_DIR / "shap_importance.png", dpi=150)
    top = imp["feature"].iloc[0]
    plot_dependence(X, sv, bundle["features"], top).savefig(FIG_DIR / "shap_top_feature.png", dpi=150)
    imp.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {RESULTS_FILE}, shap_importance.png and shap_top_feature.png")
