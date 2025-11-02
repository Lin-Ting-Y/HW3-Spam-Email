import os
import re
import json
import joblib
import streamlit as st
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_PATH = os.path.join('models', 'spam_svm.joblib')
METRICS_PATH = os.path.join('reports', 'metrics.json')
CM_FIG_PATH = os.path.join('reports', 'confusion_matrix.png')
PR_FIG_PATH = os.path.join('reports', 'pr_curve.png')
ROC_FIG_PATH = os.path.join('reports', 'roc_curve.png')
TEST_PATH = os.path.join('data', 'processed', 'test.csv')

st.set_page_config(page_title='Spam Classifier (SVM)', page_icon='📩', layout='wide')
st.title('📩 SMS Spam Classifier — SVM')
st.caption('Baseline TF‑IDF + LinearSVC (calibrated) — interactive demo')

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()

# Sidebar summary
with st.sidebar:
    st.header('Model Summary')
    if model is None:
        st.error('Model: not found')
    else:
        st.success('Model: loaded')
    if metrics:
        st.subheader('Metrics')
        st.metric('Accuracy', f"{metrics.get('accuracy', 0):.3f}")
        st.metric('F1', f"{metrics.get('f1', 0):.3f}")
        ap_val = metrics.get('average_precision', None)
        auc_val = metrics.get('roc_auc', None)
        if ap_val is not None:
            st.metric('AP', f"{ap_val:.3f}")
        if auc_val is not None:
            st.metric('ROC AUC', f"{auc_val:.3f}")
    st.caption('Artifacts expected in models/ and reports/.')

# Tabs for flow similar to example app
tab_predict, tab_explore, tab_about = st.tabs(["Predict", "Explore", "About"])

with tab_predict:
    st.subheader('Try it out')
    examples = {
        'Example (Ham)': "Hey there, are we still on for lunch at 12?",
        'Example (Spam)': "Congratulations! You won a $1000 gift card. Click here to claim now!"
    }
    ex1, ex2 = st.columns(2)
    if ex1.button('Fill Ham Example'):
        st.session_state['input_text'] = examples['Example (Ham)']
    if ex2.button('Fill Spam Example'):
        st.session_state['input_text'] = examples['Example (Spam)']

    default_text = st.session_state.get('input_text', '')
    text = st.text_area('Enter a message:', value=default_text, height=120, placeholder='e.g., Congratulations! You won a prize...')

    col1, col2 = st.columns([1,1])
    predict_clicked = col1.button('Classify')
    if col2.button('Clear Input'):
        st.session_state['input_text'] = ''
        text = ''

    if predict_clicked:
        if model is None:
            st.error('Model not found. Please run training first: python scripts/train_baseline.py')
        elif not text.strip():
            st.warning('Please enter a non-empty message.')
        else:
            # Calibrated model supports predict_proba
            proba = None
            try:
                proba = model.predict_proba([text])[0]
            except Exception:
                proba = None
            pred = model.predict([text])[0]
            st.success(f'Prediction: {pred.upper()}')
            if proba is not None and len(proba) == 2:
                classes = getattr(model.named_steps['clf'], 'classes_', None)
                if classes is not None:
                    prob_map = dict(zip(classes, proba))
                    spam_p = float(prob_map.get('spam', 0.0))
                    ham_p = float(prob_map.get('ham', 0.0))
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption('Spam probability')
                        st.progress(min(int(spam_p*100), 100))
                        st.write(f"{spam_p:.3f}")
                    with c2:
                        st.caption('Ham probability')
                        st.progress(min(int(ham_p*100), 100))
                        st.write(f"{ham_p:.3f}")

with tab_explore:
    st.subheader('Model & Evaluation')
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Accuracy', f"{metrics.get('accuracy', 0):.3f}")
        c2.metric('Precision', f"{metrics.get('precision', 0):.3f}")
        c3.metric('Recall', f"{metrics.get('recall', 0):.3f}")
        c4.metric('F1', f"{metrics.get('f1', 0):.3f}")
    else:
        st.info('No metrics found. Train the model first with scripts/train_baseline.py.')

    if os.path.exists(CM_FIG_PATH):
        st.image(CM_FIG_PATH, caption='Confusion matrix', use_container_width=True)

    # ---- Decision Threshold Explorer ----
    st.subheader('Decision Threshold')

    @st.cache_data(show_spinner=False)
    def load_test_df(path: str):
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            if not {'label', 'text'}.issubset(df.columns):
                return None
            df = df.dropna(subset=['label', 'text'])
            df['text'] = df['text'].astype(str)
            return df
        except Exception:
            return None

    threshold = st.slider('Decision threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    test_df = load_test_df(TEST_PATH)
    if model is None:
        st.info('Model not loaded. Train the model first to enable threshold exploration.')
    elif test_df is None or test_df.empty:
        st.info('Test split not found at data/processed/test.csv. Run the ingestion/training pipeline first.')
    else:
        texts = test_df['text'].tolist()
        y_true = test_df['label'].tolist()
        # Compute probabilities for the spam class
        y_scores = None
        try:
            proba = model.predict_proba(texts)
            classes = getattr(model.named_steps['clf'], 'classes_', None)
            if classes is not None and 'spam' in list(classes):
                spam_idx = list(classes).index('spam')
            else:
                spam_idx = 1
            y_scores = proba[:, spam_idx]
        except Exception:
            # As a fallback, try decision_function; map margins to [0,1] via logistic-like transform if needed.
            try:
                margins = model.decision_function(texts)
                import numpy as _np
                if getattr(margins, 'ndim', 1) > 1:
                    classes = getattr(model.named_steps['clf'], 'classes_', None)
                    spam_idx = list(classes).index('spam') if classes is not None and 'spam' in list(classes) else 1
                    margins = margins[:, spam_idx]
                # Sigmoid transform to pseudo-probabilities
                y_scores = 1.0 / (1.0 + _np.exp(-margins))
            except Exception:
                y_scores = None

        if y_scores is None:
            st.warning('Could not compute probabilities or scores from the model for thresholding.')
        else:
            # Apply threshold to produce labels
            y_pred = ['spam' if s >= threshold else 'ham' for s in y_scores]
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label='spam', zero_division=0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Accuracy (thr)', f"{acc:.3f}")
            c2.metric('Precision (thr)', f"{prec:.3f}")
            c3.metric('Recall (thr)', f"{rec:.3f}")
            c4.metric('F1 (thr)', f"{f1:.3f}")

    # PR and ROC curves (if available) — shown below threshold explorer
    pr_exists = os.path.exists(PR_FIG_PATH)
    roc_exists = os.path.exists(ROC_FIG_PATH)
    if pr_exists or roc_exists:
        st.subheader('Precision-Recall and ROC Curves')
        col_pr, col_roc = st.columns(2)
        with col_pr:
            if pr_exists:
                st.image(PR_FIG_PATH, caption='Precision-Recall Curve', use_container_width=True)
            else:
                st.info('PR curve not found. Run training to generate it.')
        with col_roc:
            if roc_exists:
                st.image(ROC_FIG_PATH, caption='ROC Curve', use_container_width=True)
            else:
                st.info('ROC curve not found. Run training to generate it.')

        # Display AP and ROC-AUC metrics if available, and a download for metrics.json
        if metrics:
            ap = metrics.get('average_precision', None)
            auc = metrics.get('roc_auc', None)
            k1, k2 = st.columns(2)
            with k1:
                if ap is not None:
                    st.metric('Average Precision (AP)', f"{ap:.3f}")
                else:
                    st.info('AP not available. Re-run training to generate PR curve.')
            with k2:
                if auc is not None:
                    st.metric('ROC AUC', f"{auc:.3f}")
                else:
                    st.info('ROC AUC not available. Re-run training to generate ROC curve.')

            st.download_button(
                label='Download metrics.json',
                data=json.dumps(metrics, indent=2),
                file_name='metrics.json',
                mime='application/json'
            )

    # ---- Top Tokens by Class (dynamic) ----
    st.subheader('Top Tokens by Class')
    top_n = st.slider('Top-N tokens', min_value=5, max_value=30, value=15)

    RAW_PATH = os.path.join('data', 'sms_spam_no_header.csv')

    @st.cache_data(show_spinner=False)
    def load_raw_df(path: str):
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, header=None, names=['label', 'text'], encoding='latin-1')
            df = df.dropna(subset=['label', 'text'])
            df['text'] = df['text'].astype(str)
            return df
        except Exception:
            return None

    def tokenize(text: str):
        # simple word tokenizer: letters + apostrophes
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]

    def compute_top_tokens(df: pd.DataFrame, label: str, k: int) -> pd.DataFrame:
        subset = df[df['label'] == label]['text'] if df is not None else pd.Series([], dtype=str)
        counter = Counter()
        for msg in subset:
            counter.update(tokenize(msg))
        most = counter.most_common(k)
        if not most:
            return pd.DataFrame({'token': [], 'count': []})
        out = pd.DataFrame(most, columns=['token', 'count'])
        out.set_index('token', inplace=True)
        return out

    raw_df = load_raw_df(RAW_PATH)
    if raw_df is None:
        st.info('Raw dataset not found at data/sms_spam_no_header.csv. Please ensure the file exists.')
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.caption('Ham — Top tokens')
            ham_df = compute_top_tokens(raw_df, 'ham', top_n)
            if not ham_df.empty:
                st.bar_chart(ham_df, use_container_width=True)
            else:
                st.write('No tokens available.')
        with c2:
            st.caption('Spam — Top tokens')
            spam_df = compute_top_tokens(raw_df, 'spam', top_n)
            if not spam_df.empty:
                st.bar_chart(spam_df, use_container_width=True)
            else:
                st.write('No tokens available.')

with tab_about:
    st.subheader('About')
    st.write('This demo uses a TF‑IDF + LinearSVC (calibrated) model trained on the SMS spam dataset. Use the Predict tab to classify messages and the Explore tab to view metrics, confusion matrix, PR/ROC curves, and top tokens by class.')
