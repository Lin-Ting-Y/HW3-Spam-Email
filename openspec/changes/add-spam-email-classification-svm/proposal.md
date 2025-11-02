## Why
Spam detection is a common and valuable capability for messaging systems. This change proposes building a spam email/sms classifier using a Support Vector Machine (SVM) as a first-pass model to establish a baseline and then iterate with improvements.

## What Changes
- ADDED: New capability `spam-classification` with an initial SVM-based implementation and dataset-driven baseline.
- ADDED: Data ingestion pipeline for the Packt Publishing SMS spam dataset.
- ADDED: Training and evaluation scripts, baseline metrics, and reproducible notebook.
 - ADDED: Streamlit application for interactive inference and visualizations (Phase 2), including deployment to Streamlit Cloud.

## Impact
- Affected specs: `spam-classification` (new capability)
- Affected code: `models/`, `data/`, `notebooks/` for experiments, and `streamlit_app.py` for the UI; CI may add CPU runners for training tasks.
