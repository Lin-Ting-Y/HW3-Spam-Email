import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)

DATA_DIR = os.path.join('data', 'processed')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
RAW_PATH = os.path.join('data', 'sms_spam_no_header.csv')
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
MODEL_PATH = os.path.join(MODELS_DIR, 'spam_svm.joblib')
METRICS_PATH = os.path.join(REPORTS_DIR, 'metrics.json')
CM_FIG_PATH = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
PR_FIG_PATH = os.path.join(REPORTS_DIR, 'pr_curve.png')
ROC_FIG_PATH = os.path.join(REPORTS_DIR, 'roc_curve.png')


def ensure_splits(test_size: float = 0.2, seed: int = 42):
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        return
    # Fallback: create splits from raw
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f'Missing processed splits and raw dataset not found: {RAW_PATH}. Run scripts/ingest_spam.py first.'
        )
    df = pd.read_csv(RAW_PATH, header=None, names=['label', 'text'], encoding='latin-1')
    df = df.dropna(subset=['label', 'text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != '']
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['label']
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_pipeline(max_features: int = 50000) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )
    base = LinearSVC()
    clf = CalibratedClassifierCV(estimator=base, cv=5)  # adds predict_proba (sklearn>=1.7 uses 'estimator')
    pipe = Pipeline([
        ('tfidf', vectorizer),
        ('clf', clf),
    ])
    return pipe


def evaluate_and_report(model: Pipeline, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='spam')

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'labels': ['ham', 'spam']
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(CM_FIG_PATH)
    plt.close()

    # Probability scores for curves (use positive class 'spam')
    # Try predict_proba; if not available, fall back to decision_function
    y_true_bin = np.array([1 if y == 'spam' else 0 for y in y_test])
    try:
        proba = model.predict_proba(X_test)
        # Determine index of 'spam' class from the calibrated classifier step
        clf = model.named_steps['clf']
        classes_ = list(getattr(clf, 'classes_', []))
        spam_idx = classes_.index('spam') if 'spam' in classes_ else 1
        y_scores = proba[:, spam_idx]
    except Exception:
        # decision_function outputs margin; good enough as a score for PR/ROC
        try:
            y_scores = model.decision_function(X_test)
            # If multi-class margins, pick column corresponding to 'spam'
            if y_scores.ndim > 1:
                clf = model.named_steps['clf']
                classes_ = list(getattr(clf, 'classes_', []))
                spam_idx = classes_.index('spam') if 'spam' in classes_ else 1
                y_scores = y_scores[:, spam_idx]
        except Exception:
            y_scores = None

    if y_scores is not None:
        # Precision-Recall curve + Average Precision (area under PR)
        prec, rec, _ = precision_recall_curve(y_true_bin, y_scores)
        ap = average_precision_score(y_true_bin, y_scores)
        plt.figure(figsize=(4.5, 3.5))
        plt.plot(rec, prec, color='purple', lw=2, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (positive: spam)')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PR_FIG_PATH)
        plt.close()

        # ROC curve + AUC
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
        auc = roc_auc_score(y_true_bin, y_scores)
        plt.figure(figsize=(4.5, 3.5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (positive: spam)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROC_FIG_PATH)
        plt.close()

        # Update metrics with AP and AUC
        metrics.update({
            'average_precision': float(ap),
            'roc_auc': float(auc),
            'pr_curve_path': PR_FIG_PATH,
            'roc_curve_path': ROC_FIG_PATH,
        })
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)

    print('Evaluation metrics:')
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Train baseline SVM spam classifier.')
    parser.add_argument('--max-features', type=int, default=50000, help='Max TF-IDF features')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    ensure_splits(test_size=args.test_size, seed=args.seed)
    train_df, test_df = load_splits()

    X_train, y_train = train_df['text'].tolist(), train_df['label'].tolist()
    X_test, y_test = test_df['text'].tolist(), test_df['label'].tolist()

    model = build_pipeline(max_features=args.max_features)
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'Saved model -> {MODEL_PATH}')

    evaluate_and_report(model, X_test, y_test)


if __name__ == '__main__':
    main()
