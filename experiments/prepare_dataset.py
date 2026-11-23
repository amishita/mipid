
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_NAME = "Necent/llm-jailbreak-prompt-injection-dataset"
TOTAL_PER_CLASS = 6000
RANDOM_STATE = 42
N_SPLITS = 5

# --------------------------------------------------
# Download dataset
# --------------------------------------------------

print("Downloading dataset from Hugging Face...")
dataset = load_dataset(DATASET_NAME)

df = dataset["train"].to_pandas()
print("Original rows:", len(df))

# --------------------------------------------------
# Prepare MIPID columns
# --------------------------------------------------

required_columns = [
    "prompt",
    "prompt_adversarial",
    "source",
    "language",
    "prompt_type",
    "category",
    "attack_technique",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

df = df[required_columns].copy()

df = df.rename(columns={
    "prompt": "text",
    "prompt_adversarial": "label"
})

df = df.dropna(subset=["text", "label"])

df["text"] = df["text"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["label"])
df = df[(df["text"] != "") & (df["label"].isin([0, 1]))]

df["label"] = df["label"].astype(int).map({
    0: "benign",
    1: "injection"
})

df["source"] = df["source"].fillna("unknown").astype(str)

# Remove exact duplicate prompts
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("\nUnique prompts:", len(df))
print("\nAvailable class distribution:")
print(df["label"].value_counts())

# --------------------------------------------------
# Create balanced 12,000-row sample
# --------------------------------------------------

benign = df[df["label"] == "benign"]
injection = df[df["label"] == "injection"]

if len(benign) < TOTAL_PER_CLASS or len(injection) < TOTAL_PER_CLASS:
    raise ValueError(
        "Insufficient examples for a balanced dataset. "
        f"Benign: {len(benign)}, Injection: {len(injection)}"
    )

sample = pd.concat([
    benign.sample(n=TOTAL_PER_CLASS, random_state=RANDOM_STATE),
    injection.sample(n=TOTAL_PER_CLASS, random_state=RANDOM_STATE)
], ignore_index=True)

sample = sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nSample created:", len(sample))

# --------------------------------------------------
# Stratified group splitting
# --------------------------------------------------

# Five folds of approximately 20% each.
# Fold 0: Test
# Fold 1: Validation
# Folds 2, 3, 4: Training

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

folds = list(splitter.split(
    X=sample["text"],
    y=sample["label"],
    groups=sample["source"]
))

test_indices = folds[0][1]
validation_indices = folds[1][1]

train_indices = [
    index
    for fold in folds[2:]
    for index in fold[1]
]

train = sample.iloc[train_indices].copy().reset_index(drop=True)
validation = sample.iloc[validation_indices].copy().reset_index(drop=True)
test = sample.iloc[test_indices].copy().reset_index(drop=True)

# --------------------------------------------------
# Validate the splits
# --------------------------------------------------

splits = {
    "TRAIN": train,
    "VALIDATION": validation,
    "TEST": test
}

for name, part in splits.items():

    if set(part["label"].unique()) != {"benign", "injection"}:
        raise ValueError(
            f"{name} does not contain both classes. "
            f"Distribution: {part['label'].value_counts().to_dict()}"
        )

# Check source separation
train_sources = set(train["source"])
validation_sources = set(validation["source"])
test_sources = set(test["source"])

assert train_sources.isdisjoint(validation_sources)
assert train_sources.isdisjoint(test_sources)
assert validation_sources.isdisjoint(test_sources)

# Check exact prompt overlap
train_texts = set(train["text"])
validation_texts = set(validation["text"])
test_texts = set(test["text"])

assert train_texts.isdisjoint(validation_texts)
assert train_texts.isdisjoint(test_texts)
assert validation_texts.isdisjoint(test_texts)

# --------------------------------------------------
# Save datasets
# --------------------------------------------------

sample.to_csv(OUTPUT_DIR / "master_12000.csv", index=False)

train.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test.to_csv(OUTPUT_DIR / "test.csv", index=False)

# --------------------------------------------------
# Print final report
# --------------------------------------------------

print("\n" + "=" * 45)
print("       MIPID DATASET PREPARATION REPORT")
print("=" * 45)

for name, part in splits.items():

    print(f"\n{name}")
    print("-" * 30)
    print("Total rows:", len(part))
    print("Class distribution:")
    print(part["label"].value_counts().to_dict())
    print("Unique sources:", part["source"].nunique())

print("\n" + "=" * 45)
print("ALL VALIDATION CHECKS PASSED")
print("=" * 45)

print("No source overlap.")
print("No exact prompt overlap.")
print("Both classes present in every split.")

print("\nFiles saved successfully in:", OUTPUT_DIR.resolve())
