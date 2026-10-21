
import pandas as pd
from mipid.detector import predict_prompts
from mipid.explanations import attack_category


def analyze_batch(model, dataframe):
    """Validate and classify a batch of prompts."""

    if "text" not in dataframe.columns:
        raise ValueError("CSV must contain a 'text' column.")

    results = dataframe.dropna(subset=["text"]).copy()
    prompts = results["text"].astype(str).tolist()

    predictions, scores = predict_prompts(model, prompts)

    results["prediction"] = predictions
    results["injection_score"] = scores.round(4)

    results["risk_level"] = results["injection_score"].apply(
        lambda score: (
            "High" if score >= 0.75
            else "Medium" if score >= 0.40
            else "Low"
        )
    )

    details = results["text"].apply(attack_category)

    results["possible_attack_category"] = details.apply(
        lambda item: item[0]
    )
    results["matched_indicators"] = details.apply(
        lambda item: ", ".join(item[1]) if item[1] else "None detected"
    )

    return results
