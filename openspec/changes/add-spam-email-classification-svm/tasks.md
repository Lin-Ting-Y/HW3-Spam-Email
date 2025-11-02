## 1. Implementation (Phase 1 — Baseline)
- [x] 1.1 Create `data/` folder and download dataset: https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv
- [x] 1.2 Build ingestion script to parse CSV and produce train/test splits
- [x] 1.3 Implement baseline SVM training script (scikit-learn) and save model artifacts
- [x] 1.4 Evaluate baseline and record metrics (accuracy, precision, recall, F1)
  

## 2. Implementation (Phase 2 — Improvements)
- [ ] 2.1 Add text preprocessing improvements: TF-IDF, n-grams, stop-word handling
- [ ] 2.2 Experiment with class-weighting and hyperparameter tuning (GridSearchCV)
- [ ] 2.3 Add more datasets (optional) and cross-validation
- [ ] 2.4 Build a Streamlit app `streamlit_app.py` that accepts user text input and shows the classification result (spam/ham) with confidence
- [ ] 2.5 Add visualizations in the Streamlit app (e.g., accuracy/F1 summary, confusion matrix, and optional word cloud)

## 3. Validation
- [ ] 3.1 Add unit tests for data pipeline and model inference
- [ ] 3.2 Add an evaluation report under `reports/` with baseline and improved metrics

## 4. Deployment
- [x] 4.1 Add `requirements.txt` suitable for Streamlit deployment (include streamlit, scikit-learn, pandas, numpy, joblib, wordcloud (optional))
- [ ] 4.2 Deploy the app to Streamlit Cloud and verify it loads the model and performs inference interactively

## 5. Source Control and GitHub Upload
- [ ] 5.1 Initialize a local Git repo (if needed) and create an initial commit covering current project files
- [ ] 5.2 Create a GitHub repository and add it as the `origin` remote
- [ ] 5.3 Push the main branch to GitHub (`git push -u origin main`) and verify files are visible on GitHub
- [ ] 5.4 Add/update `.gitignore` to exclude local artifacts (e.g., `data/processed/`, `models/`, `__pycache__/`) and commit
- [ ] 5.5 (Optional) Connect Streamlit Cloud to the GitHub repo for automatic deploys
