# SMS Spam Classifier — SVM (Streamlit)

This project trains a baseline SMS spam classifier (TF‑IDF + LinearSVC with calibrated probabilities) and provides a Streamlit app for interactive inference and visuals.

Live demo:
- https://hw3-spam-email.streamlit.app/

## Setup (Windows PowerShell)

```powershell
# (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Or using Conda (recommended on Windows):

```powershell
conda create -n newenv python=3.12 -y
conda run -n newenv python -m pip install -r requirements.txt
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
- `reports/metrics.json` — accuracy, precision, recall, F1, AP, ROC‑AUC
- `reports/confusion_matrix.png`
- `reports/pr_curve.png`
- `reports/roc_curve.png`

## Run Streamlit App
```powershell
streamlit run streamlit_app.py
```

With Conda:

```powershell
C:\Users\<you>\anaconda3\Scripts\conda.exe run -n newenv python -m streamlit run .\streamlit_app.py --server.port 8501
```

VS Code one-click task:
- Run Task → "Run Streamlit (newenv, 8501)" (starts in background)

App highlights:
- Predict tab: classify a message; shows calibrated probabilities for ham/spam
- Explore tab:
	- Baseline metrics and confusion matrix (from reports/)
	- Decision threshold slider that re-computes Accuracy/Precision/Recall/F1 on the test split
	- PR and ROC curves (from reports/)
	- Top tokens by class charts

## Notes
- The app expects the trained model and metrics to exist; run the training step first.
- You can tweak TF‑IDF (ngrams, min_df) or classifier settings in `scripts/train_baseline.py`.
- For Streamlit Cloud, keep `requirements.txt` and `streamlit_app.py` at the repo root. The demo above is deployed from this repo.
