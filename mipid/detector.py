
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def train_detector(data_path):
    """Train the baseline detector from a labeled CSV."""

    data = pd.read_csv(data_path)

    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset must contain columns: {required}")

    train_texts, _, train_labels, _ = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "svm",
            LinearSVC(class_weight="balanced", random_state=42),
        ),
    ])

    model = CalibratedClassifierCV(pipeline, cv=3)
    model.fit(train_texts, train_labels)

    return model


def predict_prompts(model, prompts):
    """Return predictions and estimated injection scores."""

    predictions = model.predict(prompts)
    probabilities = model.predict_proba(prompts)

    injection_index = list(model.classes_).index("injection")
    scores = probabilities[:, injection_index]

    return predictions, scores
