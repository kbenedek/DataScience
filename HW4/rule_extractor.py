from argparse import ArgumentParser
from itertools import combinations
from json import dump

import numpy as np
import pandas as pd


def subgraph_frequencies(
    moss_matrix: np.ndarray, C: pd.DataFrame, max_X: int = 2
) -> dict:
    frequencies = {}
    subgraph_combinations = []

    for size in range(1, max_X + 1):
        subgraph_combinations.extend(combinations(range(moss_matrix.shape[0]), size))

    for combo in subgraph_combinations:
        combo_key = tuple(sorted(combo))
        X_mask = np.all(moss_matrix[list(combo), :], axis=0)
        fr_X = np.sum(X_mask)

        frequencies[combo_key] = {"fr_X": fr_X}

        for cancer_type in range(1, 4):
            cancer_column = C[f"anti_cancer_{cancer_type}"].values.astype(bool)
            fr_XC = np.sum(X_mask & cancer_column)
            frequencies[combo_key][f"fr_XC{cancer_type}"] = fr_XC

    return frequencies


def get_subgraphs(
    moss_file: str, n_molecules: int = 47, n_subgraphs: int = 88
) -> np.ndarray:
    moss = np.zeros((n_subgraphs, n_molecules), dtype=bool)

    with open(moss_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            sub_id, sup_ids = line.strip().split(":")
            sup_ids = sup_ids.split(",")

            if not sub_id.isdigit() or not all(sup_id.isdigit() for sup_id in sup_ids):
                continue

            sub_id = int(sub_id) - 1
            sub_ids = [int(sup_id) - 1 for sup_id in sup_ids]

            moss[sub_id, sub_ids] = True
    return moss


def xlogx(x):
    return 0 if x <= 0 else x * np.log2(x)


def mutual_information(P_X, P_XC, P_C, total_count):
    P_X_notC = P_X - P_XC
    P_notX_C = P_C - P_XC
    P_notX_notC = 1 - (P_X + P_C - P_XC)

    MI = (
        xlogx(P_XC)
        + xlogx(P_X_notC)
        + xlogx(P_notX_C)
        + xlogx(P_notX_notC)
        - xlogx(P_X)
        - xlogx(1 - P_X)
        - xlogx(P_C)
        - xlogx(1 - P_C)
    )

    return total_count * MI


def compute_P_C(subgraphs: np.ndarray, C: str):
    P_C = {"P_C1": 0, "P_C2": 0, "P_C3": 0}

    for row in subgraphs:
        row_sub = C.iloc[row][["anti_cancer_1", "anti_cancer_2", "anti_cancer_3"]]
        for cancer_type in range(1, 4):
            P_C[f"P_C{cancer_type}"] += (
                1 if any(row_sub[f"anti_cancer_{cancer_type}"]) else 0
            )

    for cancer_type in range(1, 4):
        P_C[f"P_C{cancer_type}"] /= subgraphs.shape[0]

    return P_C


def extract_significant_rules(
    moss_matrix: np.ndarray, C: pd.DataFrame, threshold: float = 0.01
):
    n_subgraphs = moss_matrix.shape[1]
    frequencies = subgraph_frequencies(moss_matrix, C, max_X=2)
    P_C = compute_P_C(moss_matrix, C)

    rules = {1: [], 2: []}

    for X, freq in frequencies.items():
        for cancer_type in range(1, 4):
            freq_key = f"fr_XC{cancer_type}"
            P_X = freq["fr_X"] / n_subgraphs
            P_XC = freq[freq_key] / n_subgraphs
            MI = mutual_information(P_X, P_XC, P_C[f"P_C{cancer_type}"], n_subgraphs)
            if MI > threshold:
                rules[len(X)].append((X, cancer_type, MI))

    return rules


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract significant rules from MoSS output.")
    parser.add_argument(
        "--moss_file",
        type=str,
        default="molecule.db.moss",
        help="Path to the MoSS output file.",
    )
    parser.add_argument(
        "--molecule_file",
        type=str,
        default="molecule.csv",
        help="Path to the molecule CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Mutual information threshold for rule significance.",
    )

    args = parser.parse_args()

    C = pd.read_csv(args.molecule_file, sep=";")
    moss_matrix = get_subgraphs(args.moss_file)
    significant_rules = extract_significant_rules(
        moss_matrix, C, threshold=args.threshold
    )

    for size, rules in significant_rules.items():
        print(f"Significant rules of size {size}:")
        for rule in rules:
            X, cancer_type, MI = rule
            print(f"  Rule: {X} -> anti_cancer_{cancer_type}, MI: {MI:.4f}")

    dump(significant_rules, open("significant_rules.json", "w"), indent=4)
    print("Significant rules saved to 'significant_rules.json'.")
