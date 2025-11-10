from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INPUT_FILE = "./birds2025ext.csv"
OUTPUT_FILE = "birdstrans.txt"

TAXONOMY_MAP = {
    "Laridae": "Lari",
    "Sternidae": "Lari",
    "Scolopacidae": "Charadrii",
    "Charadriidae": "Charadrii",
    "dabbling ducks": "Anatinae",
    "diving ducks": "Anatinae",
}

BINARY_COLS = [
    "diver",
    "long-billed",
    "webbed-feet",
    "long-feet",
    "wading-bird",
    "plunge-dives",
]
MULTI_VALUE_COLS = ["back", "belly", "ftype", "billcol", "legcol", "incub", "ccare"]


def parse_interval(val: Any) -> float:
    """Convert interval string 'X-Y' to mean, or return float value."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    if "-" in s:
        try:
            parts = s.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, IndexError):
            pass
    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_prefixed_features(text: str, prefix: str) -> list[str]:
    """Split multi-element field and return prefixed feature list."""
    if pd.isna(text):
        return []
    elements = str(text).replace(",", " ").split()
    return [f"{prefix}_{e.lower().replace(' ', '_')}" for e in elements if e]


def discretize_to_extremes(df: pd.DataFrame, col: str, feature_name: str) -> pd.Series:
    """Return discretized features for extreme values (Q1/Q3)."""
    q1, q3 = df[col].quantile([0.25, 0.75])

    def categorize(val):
        if pd.isna(val):
            return None
        if val <= q1:
            return f"low_{feature_name}"
        elif val >= q3:
            return f"high_{feature_name}"
        return None

    return df[col].apply(categorize)


def extract_row_features(row: pd.Series, discretized_features: list[str]) -> list[str]:
    """Extract all features for a single bird species."""
    features = []

    for feat in discretized_features:
        if pd.notna(row[feat]):
            features.append(row[feat])

    group = row.get("group")
    if pd.notna(group):
        group_clean = str(group).lower().replace(" ", "_")
        features.append(f"group_{group_clean}")
        if group in TAXONOMY_MAP:
            features.append(f"group_{TAXONOMY_MAP[group].lower()}")

    features.extend(extract_prefixed_features(row.get("diet"), "eats"))
    features.extend(extract_prefixed_features(row.get("biotope"), "livesin"))

    for col in BINARY_COLS:
        if row.get(col) == "Yes":
            features.append(col.replace("-", "_"))

    sim = row.get("sim")
    if sim == "Yes":
        features.append("genders_similar")
    elif sim == "No":
        features.append("genders_dissimilar")

    for col in MULTI_VALUE_COLS:
        val = row.get(col)
        if pd.notna(val):
            parts = str(val).lower().replace("-", " ").split()
            features.extend(f"{col}_{p}" for p in parts)

    arrives = row.get("mean_arrives")
    leaves = row.get("mean_leaves")

    if pd.notna(arrives) and arrives <= 4:
        features.append("migrates_arrives_early")
    if pd.notna(leaves) and leaves >= 10:
        features.append("migrates_leaves_late")

    return sorted(set(features))


def main():
    if not Path(INPUT_FILE).exists():
        print(f"Error: Could not find {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, sep=";", index_col="species")

    interval_cols = ["length", "wspan", "weight", "eggs", "arrives", "leaves"]
    for col in interval_cols:
        if col in df.columns:
            df[f"mean_{col}"] = df[col].apply(parse_interval)

    df["BMI"] = (df["mean_weight"] / 1000) / (df["mean_length"] / 100) ** 2
    df["WSI"] = df["mean_wspan"] / df["mean_length"]

    discretized_cols = []
    for col, name in [("mean_eggs", "eggs"), ("BMI", "bmi"), ("WSI", "wsi")]:
        if col in df.columns:
            feat_col = f"disc_{name}"
            df[feat_col] = discretize_to_extremes(df, col, name)
            discretized_cols.append(feat_col)

    transactions = [
        extract_row_features(row, discretized_cols) for _, row in df.iterrows()
    ]

    with open(OUTPUT_FILE, "w") as f:
        for features in transactions:
            f.write(" ".join(features) + "\n")

    print(f"  Feature extraction complete. Saved to {OUTPUT_FILE}")
    print(
        f"  Processed {len(transactions)} species with {sum(len(t) for t in transactions)} total features"
    )


if __name__ == "__main__":
    main()
