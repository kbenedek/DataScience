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


def load_subgraph_smiles(sub_file: str) -> dict:
    """
    Load SMILES notation for each subgraph from .sub file.

    Args:
        sub_file: Path to .sub file (e.g., train.sub)

    Returns:
        Dictionary mapping subgraph ID (0-indexed) to SMILES string
    """
    smiles_map = {}
    df = pd.read_csv(sub_file)

    for _, row in df.iterrows():
        sub_id = int(row["id"]) - 1  # Convert to 0-indexed
        smiles = row["description"]
        smiles_map[sub_id] = smiles

    return smiles_map


def get_subgraphs(
    moss_file: str, n_molecules: int = 46, n_subgraphs: int = 88
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


def mutual_information(P_X, P_XC, P_C):
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

    return MI


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
            MI = mutual_information(P_X, P_XC, P_C[f"P_C{cancer_type}"])
            if MI > threshold:
                rules[len(X)].append((X, cancer_type, MI))

    return rules


"""MI_C = \log \frac{P(\mathbf{X})^{P(\mathbf{X})} P(\mathbf{X}\mathbf{Q}C)^{P(\mathbf{X}\mathbf{Q}C)} P(\mathbf{X}\mathbf{Q}\neg C)^{P(\mathbf{X}\mathbf{Q}\neg C)} P(\mathbf{X}\neg \mathbf{Q}C)^{P(\mathbf{X}\neg \mathbf{Q}C)} P(\mathbf{X}\neg \mathbf{Q}\neg C)^{P(\mathbf{X}\neg \mathbf{Q}\neg C)}}{P(\mathbf{X}\mathbf{Q})^{P(\mathbf{X}\mathbf{Q})} P(\mathbf{X}\neg \mathbf{Q})^{P(\mathbf{X}\neg \mathbf{Q})} P(\mathbf{X}C)^{P(\mathbf{X}C)} P(\mathbf{X}\neg C)^{P(\mathbf{X}\neg C)}}"""


def conditional_mutual_information(
    frequencies: dict,
    X: tuple,
    Q: int,
    cancer_type: int,
    n_subgraphs: int,
    moss_matrix: np.ndarray,
    C: pd.DataFrame,
) -> float:
    X_set = set(X)
    XQ = tuple(sorted(X_set.union({Q})))

    fr_XQ = frequencies.get(XQ, {}).get("fr_X", 0)
    fr_XQC = frequencies.get(XQ, {}).get(f"fr_XC{cancer_type}", 0)

    X_mask = np.all(moss_matrix[list(X), :], axis=0)
    Q_mask = moss_matrix[Q, :]
    X_notQ_mask = X_mask & ~Q_mask

    fr_X_notQ = np.sum(X_notQ_mask)
    cancer_column = C[f"anti_cancer_{cancer_type}"].values.astype(bool)
    fr_X_notQC = np.sum(X_notQ_mask & cancer_column)

    fr_X = frequencies.get(X, {}).get("fr_X", 0)
    fr_XC = frequencies.get(X, {}).get(f"fr_XC{cancer_type}", 0)

    P_X = fr_X / n_subgraphs
    P_XC = fr_XC / n_subgraphs
    P_XQ = fr_XQ / n_subgraphs
    P_XQC = fr_XQC / n_subgraphs
    P_X_notQ = fr_X_notQ / n_subgraphs
    P_X_notQC = fr_X_notQC / n_subgraphs

    numerator = (
        xlogx(P_X)
        + xlogx(P_XQC)
        + xlogx(P_X_notQC)
        + xlogx(P_XQ - P_XQC)  # P(XQ\neg C)
        + xlogx(P_X_notQ - P_X_notQC)  # P(X\neg Q\neg C)
    )

    denominator = xlogx(P_XQ) + xlogx(P_X_notQ) + xlogx(P_XC) + xlogx(P_X - P_XC)

    MI_C = numerator - denominator
    return MI_C


def compute_confidence(
    frequencies: dict,
    X: tuple,
    cancer_type: int,
) -> float:
    fr_X = frequencies.get(X, {}).get("fr_X", 1)
    fr_XC = frequencies.get(X, {}).get(f"fr_XC{cancer_type}", 0)

    if fr_X == 0:
        return 0

    return fr_XC / fr_X


def prune_rules(
    rules: dict, frequencies: dict, n_subgraphs: int, mi_c_threshold: float = 0.01
) -> dict:
    pruned_rules = {1: [], 2: []}

    pruned_rules[1] = rules[1].copy()

    for XY, cancer_type, MI in rules[2]:
        should_prune = False

        for element in XY:
            Y = (element,)

            confidence_Y = compute_confidence(frequencies, Y, cancer_type)
            confidence_XY = compute_confidence(frequencies, XY, cancer_type)

            if confidence_Y >= confidence_XY:
                should_prune = True
                break

            mi_c = conditional_mutual_information(
                frequencies,
                (XY[0] if XY[0] != element else XY[1],),
                element,
                cancer_type,
                n_subgraphs,
                moss_matrix,  # Add this parameter
                C,  # Add this parameter
            )

            if mi_c <= mi_c_threshold:
                should_prune = True
                break

        if not should_prune:
            pruned_rules[2].append((XY, cancer_type, MI))

    return pruned_rules


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--moss_file",
        type=str,
        default="train.id",
        help="Path to training MOSS output file",
    )
    parser.add_argument(
        "--sub_file",
        type=str,
        default="train.sub",
        help="Path to training MOSS subgraph file with SMILES",
    )
    parser.add_argument(
        "--cancer_data",
        type=str,
        default="molecule_train.csv",
        help="Path to training cancer data CSV",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="significant_rules.json",
        help="Path to output JSON file",
    )
    parser.add_argument("--mi_threshold", type=float, default=0.05)
    parser.add_argument("--mi_c_threshold", type=float, default=0.05)

    args = parser.parse_args()

    # Load subgraph SMILES mapping
    print("Loading subgraph SMILES...")
    smiles_map = load_subgraph_smiles(args.sub_file)

    moss_matrix = get_subgraphs(args.moss_file)
    C = pd.read_csv(args.cancer_data, sep=",")

    frequencies = subgraph_frequencies(moss_matrix, C, max_X=2)
    rules = extract_significant_rules(moss_matrix, C, threshold=args.mi_threshold)
    pruned_rules = prune_rules(
        rules, frequencies, moss_matrix.shape[1], mi_c_threshold=args.mi_c_threshold
    )

    for rule_size, rule_list in pruned_rules.items():
        print(f"Rules of size {rule_size}:")
        for rule in rule_list:
            print(f"   {rule[0]} -> cancer_type_{rule[1]} with MI {rule[2]:.4f}")

    # Save rules and SMILES mapping together
    output_data = {
        "smiles_map": {str(k): v for k, v in smiles_map.items()},
        "rules": pruned_rules,
    }

    with open(args.output_file, "w") as f:
        dump(output_data, f, indent=4)
