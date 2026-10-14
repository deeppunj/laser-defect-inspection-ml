# app.py
# Phase 8: my interactive dashboard. I start it from the project folder with:
#     streamlit run app.py
# It opens in my browser. I pick a weld and see, frame by frame, the model's
# probability of a defect next to the true label and the thermal image.

import joblib
import matplotlib.pyplot as plt
import streamlit as st

from laser_defect.dashboard import first_alarm, predict_weld, weld_summary
from laser_defect.data import load_dataset
from laser_defect.features import load_features
from laser_defect.model import MODEL_FILE

st.set_page_config(page_title="Laser weld defect inspection", layout="wide")
st.title("Laser weld defect inspection")
st.caption("Random Forest on LWIR thermal frames (Walther et al. 2025, CC BY 4.0). "
           "The model only uses the current frame, earlier frames and the weld start.")


@st.cache_resource      # I load the model and data only once, not on every click
def load_all():
    X, table = load_dataset()
    return joblib.load(MODEL_FILE), load_features(), X, table


bundle, df, X, table = load_all()
weld = st.sidebar.selectbox("Weld", sorted(df["weld"].unique()))
w = predict_weld(bundle, df, weld)

# Top row: four simple numbers about this weld.
s = weld_summary(w)
alarm, truth = first_alarm(w)
c = st.columns(4)
c[0].metric("Frames", s["frames"])
c[1].metric("Agreement with label", f"{s['agreement']:.0%}")
c[2].metric("First true unsound frame", truth if truth is not None else "none")
c[3].metric("First model alarm", alarm if alarm is not None else "none")

# Chart: probability of Unsound over time; the shaded area is the true unsound part.
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(w["frame"], w["prob_unsound"], label="model: probability of Unsound")
ax.fill_between(w["frame"], 0, w["y"], alpha=0.2, color="red", label="true label: Unsound")
ax.axhline(bundle["threshold"], ls="--", color="grey", label=f"threshold {bundle['threshold']}")
ax.set_xlabel("frame (30 per second)")
ax.set_ylim(0, 1)
ax.legend(loc="upper left")
st.pyplot(fig)

# Thermal image of one chosen frame of this weld.
frame = st.slider("Frame", int(w["frame"].min()), int(w["frame"].max()), int(w["frame"].min()))
idx = table.index[(table["weld"] == weld) & (table["frame"] == frame)]
if len(idx):
    row = w[w["frame"] == frame].iloc[0]
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(X[idx[0]], cmap="inferno")
    ax2.set_axis_off()
    st.pyplot(fig2, use_container_width=False)
    st.write(f"True label: **{row['label']}**, model probability of Unsound: **{row['prob_unsound']:.2f}**")
