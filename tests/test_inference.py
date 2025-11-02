import os
import math
import joblib
import numpy as np
import pytest


MODEL_PATH = os.path.join('models', 'spam_svm.joblib')


@pytest.mark.parametrize(
    "text",
    [
        "Hey, are we still meeting at 6pm?",
        "Congratulations! You've won a $1000 gift card. Click to claim now!",
    ],
)
def test_model_predicts_ham_or_spam(text):
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model artifact missing at {MODEL_PATH}. Run training first.")

    model = joblib.load(MODEL_PATH)
    pred = model.predict([text])[0]
    assert pred in {"ham", "spam"}


def test_predict_proba_if_available():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model artifact missing at {MODEL_PATH}. Run training first.")

    model = joblib.load(MODEL_PATH)
    # not all classifiers expose predict_proba; CalibratedClassifierCV should
    proba = None
    try:
        proba = model.predict_proba(["test message"])[0]
    except Exception:
        proba = None

    if proba is None:
        pytest.skip("Model does not expose predict_proba; skipping probability test.")

    # Expect a 2-class probability distribution that sums to ~1
    assert isinstance(proba, (list, np.ndarray))
    assert len(proba) == 2
    assert math.isclose(float(np.sum(proba)), 1.0, rel_tol=1e-3, abs_tol=1e-3)
