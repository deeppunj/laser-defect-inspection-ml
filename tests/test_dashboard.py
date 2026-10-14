# test_dashboard.py
# Phase 7/8: I test the dashboard calculations, and (if real data and the model
# exist on this computer) that the Streamlit app starts without errors.

from pathlib import Path

import pytest

from laser_defect.dashboard import first_alarm, predict_weld, weld_summary
from laser_defect.model import train_final
from tests_helpers import fake_features


def test_weld_predictions_and_summary():
    df = fake_features(signal=True)
    df["label"] = df["y"].map({0: "Sound", 1: "Unsound"})
    bundle = train_final(df, n_folds=3)
    w = predict_weld(bundle, df, "001-000")
    s = weld_summary(w)
    assert s["frames"] == 20 and s["true_unsound"] == 10
    assert s["agreement"] > 0.8
    alarm, truth = first_alarm(w)
    assert truth == 10 and alarm is not None


@pytest.mark.skipif(not Path("models/defect_classifier.joblib").exists(),
                    reason="run python -m laser_defect.model first")
def test_app_starts_without_errors():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
