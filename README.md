# Laser Weld Defect Inspection with Machine Learning

**Status: in active development (phase 0 of 10 complete).** This README is updated as each phase is finished. Results are only reported once they exist.

## Goal
Detect when a laser beam weld turns from **sound** to **unsound** (a discontinuity forms as the metal sheets move apart), using in-process sensor data from real welding experiments, and explain which physical signals drive the prediction.

## Data
Real laboratory data: 100 recorded laser butt welds at two welding speeds (1 m/min and 5 m/min).
- **Long-wave infrared (LWIR) camera**: 32 × 32 px thermal frames at 30 Hz, labelled per frame as *Sound*, *Unsound* or *Beginning* (97 welds; 10,884 sound and 5,486 unsound frames).
- **Inductive probes**: three displacement sensors (mm) at 1,000 Hz tracking sheet movement.

Source: Walther D. et al. (2025), *Dataset for weld seam analysis and discontinuity prediction in laser beam welding scenarios*, Data in Brief. [doi:10.6084/m9.figshare.27877413.v1](https://doi.org/10.6084/m9.figshare.27877413.v1), licence CC BY 4.0. The data is not stored in this repository; the download script fetches and checks it.

## Plan
| Phase | Content | Status |
|---|---|---|
| 0 | Project setup, data download with integrity check, first tests | Done |
| 1 | Load and validate thermal frames and labels | Next |
| 2 | Exploratory analysis notebook | Planned |
| 3 | Physics-motivated features from thermal frames | Planned |
| 4 | Baseline model, split by weld to avoid leakage | Planned |
| 5 | Random Forest, cross-validation, threshold choice | Planned |
| 6 | Model explanation with SHAP | Planned |
| 7 | Test suite for data, features and pipeline | Planned |
| 8 | Streamlit dashboard | Planned |
| 9 | Final results, reproducibility, limitations | Planned |

## Run it
```bash
pip install -e .
python -m laser_defect.download   # downloads ~90 MB into data/ and checks its size
pytest -v
```

## Author
Deep Punj, PhD: optical engineer and physicist moving into data science. [LinkedIn](https://linkedin.com/in/deeppunj)
