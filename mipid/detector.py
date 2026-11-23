
from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "best_model.joblib"
)


def load_detector():
    """Load the saved trained model and TF-IDF vectorizer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_prompts(artifact, prompts):
    """Predict labels and return a display-only injection score."""

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    features = vectorizer.transform(prompts)
    predictions = model.predict(features)

    # Convert SVM decision margins to a 0–1 display score.
    # This is NOT a calibrated probability.
    margins = model.decision_function(features)
    scores = 1 / (1 + np.exp(-np.clip(margins, -50, 50)))

    return predictions, scores
