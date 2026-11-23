
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# LOAD ORIGINAL DATASET
# --------------------------------------------------

print("Loading original Hugging Face dataset...")

dataset = load_dataset(
    "Necent/llm-jailbreak-prompt-injection-dataset"
)

df = dataset["train"].to_pandas()

print("Original dataset rows:", len(df))

# No sampling, deduplication, or class balancing.
X = df["prompt"].fillna("").astype(str)

y = df["prompt_adversarial"].map({
    0: "benign",
    1: "injection",
})

if y.isna().any():
    raise ValueError("Unexpected labels in prompt_adversarial.")

print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# STRATIFIED 70/15/15 SPLIT
# --------------------------------------------------

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.1764705882,
    random_state=RANDOM_STATE,
    stratify=y_train_val,
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# --------------------------------------------------
# CLASSIFIERS
# --------------------------------------------------

models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

print("\nFitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_features=300000,
    sublinear_tf=True,
    dtype="float32",
)

X_train_vec = vectorizer.fit_transform(X_train)

print("Transforming validation data...")
X_val_vec = vectorizer.transform(X_val)

print("Transforming test data...")
X_test_vec = vectorizer.transform(X_test)

print("Feature count:", len(vectorizer.vocabulary_))

# --------------------------------------------------
# TRAIN AND VALIDATE
# --------------------------------------------------

validation_results = []
trained_models = {}

for name, classifier in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    start = time.time()

    classifier.fit(X_train_vec, y_train)

    predictions = classifier.predict(X_val_vec)

    elapsed = time.time() - start

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, pos_label="injection"
        ),
        "Recall": recall_score(
            y_val, predictions, pos_label="injection"
        ),
        "F1": f1_score(
            y_val, predictions, pos_label="injection"
        ),
        "Training Time (seconds)": round(elapsed, 2),
    }

    validation_results.append(metrics)
    trained_models[name] = classifier

    # Save results immediately after each model.
    pd.DataFrame(validation_results).to_csv(
        OUTPUT_DIR / "validation_comparison.csv",
        index=False,
    )

    print(metrics)

# --------------------------------------------------
# SELECT BEST MODEL USING VALIDATION F1
# --------------------------------------------------

validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values(
    by="F1",
    ascending=False,
)

validation_df.to_csv(
    OUTPUT_DIR / "validation_comparison.csv",
    index=False,
)

best_name = validation_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nSelected model:", best_name)

# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating selected model on untouched test set...")

test_predictions = best_model.predict(X_test_vec)

test_metrics = {
    "Model": best_name,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, pos_label="injection"
    ),
    "Recall": recall_score(
        y_test, test_predictions, pos_label="injection"
    ),
    "F1": f1_score(
        y_test, test_predictions, pos_label="injection"
    ),
}

print("\nFINAL TEST RESULTS")
print(test_metrics)

print("\nClassification Report")
print(classification_report(y_test, test_predictions))

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=["benign", "injection"],
)

pd.DataFrame(
    cm,
    index=["Actual benign", "Actual injection"],
    columns=["Predicted benign", "Predicted injection"],
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

# --------------------------------------------------
# SAVE FINAL RESULTS
# --------------------------------------------------

pd.DataFrame([test_metrics]).to_csv(
    OUTPUT_DIR / "final_test_results.csv",
    index=False,
)

# --------------------------------------------------
# SAVE MODEL AND VECTORIZER
# --------------------------------------------------

artifact = {
    "model": best_model,
    "vectorizer": vectorizer,
    "model_name": best_name,
    "classes": list(best_model.classes_),
}

joblib.dump(
    artifact,
    OUTPUT_DIR / "best_model.joblib",
    compress=3,
)

with open(OUTPUT_DIR / "model_info.json", "w") as file:
    json.dump(test_metrics, file, indent=4)

# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n" + "=" * 50)
print("EXPERIMENT COMPLETE")
print("Winning model:", best_name)
print("Saved: experiments/best_model.joblib")
print("Saved: experiments/validation_comparison.csv")
print("Saved: experiments/final_test_results.csv")
print("Saved: experiments/confusion_matrix.csv")
print("=" * 50)
