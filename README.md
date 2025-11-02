# SMS Spam Classifier — SVM (Streamlit)

This project trains a baseline SMS spam classifier (TF‑IDF + LinearSVC with calibrated probabilities) and provides a Streamlit app for interactive inference and basic visuals.

## Setup (Windows PowerShell)

```powershell
# (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Data
- Place the raw dataset at `data/sms_spam_no_header.csv` (already present in this repo).

## Ingest & Train
```powershell
# Create train/test splits
python scripts/ingest_spam.py

# Train baseline model and generate reports
python scripts/train_baseline.py
```
Artifacts:
- `models/spam_svm.joblib` — trained model (Pipeline: TF‑IDF + Calibrated LinearSVC)
- `reports/metrics.json` — accuracy/precision/recall/F1
- `reports/confusion_matrix.png`

## Run Streamlit App
```powershell
streamlit run streamlit_app.py
```

## Notes
- The app expects the trained model and metrics to exist; run the training step first.
- You can tweak TF‑IDF (ngrams, min_df) or classifier settings in `scripts/train_baseline.py`.
- For Streamlit Cloud, keep `requirements.txt` and `streamlit_app.py` at the repo root.
