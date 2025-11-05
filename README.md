# SMS Spam Classifier — SVM (Streamlit)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hw3-spam-email.streamlit.app/)

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

## Evaluation

The trained baseline SVM was evaluated on the held-out test split. The numbers below are taken from `reports/metrics.json` produced by `scripts/train_baseline.py`.

- Accuracy: 0.9856502242152466 (98.565%)
- Precision (spam): 0.9523809523809523 (95.24%)
- Recall (spam): 0.9395973154362416 (93.96%)
- F1 (spam): 0.9459459459459459 (94.59%)
- Average Precision (AP): 0.9783482160876619
- ROC AUC: 0.990203843428238

Confusion matrix and curves:

- Confusion matrix image: `reports/confusion_matrix.png`
- Precision-Recall curve: `reports/pr_curve.png` (path in `reports/metrics.json`)
- ROC curve: `reports/roc_curve.png` (path in `reports/metrics.json`)

Notes on these results:

- The dataset is imbalanced (many more ham messages than spam), so overall accuracy is high; the precision, recall and F1 for the spam class are more informative about spam-detection quality.
- Precision ~= 95% means about 5% of predicted spam messages are false positives. Recall ~= 94% means the model finds most spam messages.

How to reproduce the evaluation

1. Ensure the raw dataset is present: `data/sms_spam_no_header.csv`.
2. Create train/test splits and train the model:

```powershell
python scripts/ingest_spam.py
python scripts/train_baseline.py
```

3. The training script will save metrics and figures to the `reports/` folder (see `reports/metrics.json`). You can open the confusion matrix and curves from that folder.

If you'd like, I can add a short script that loads `models/spam_svm.joblib` and prints the same metrics (or computes a confusion-matrix table) for easier CLI inspection.

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
