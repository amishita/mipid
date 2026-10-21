
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(y_true, y_pred):
    """Calculate binary classification metrics."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["injection"],
        average="binary",
        pos_label="injection",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=["benign", "injection"],
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=["benign", "injection"],
            zero_division=0,
        ),
    }
