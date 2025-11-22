import json
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem


def load_subgraph_smiles(sub_file: str) -> dict:
    """Load SMILES notation for each subgraph from .sub file."""
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
    """Extract subgraph presence matrix from MOSS output file."""
    moss = np.zeros((n_subgraphs, n_molecules), dtype=bool)

    with open(moss_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split(":")
            if len(parts) != 2:
                continue

            sub_id, sup_ids = parts
            sup_ids = sup_ids.split(",")

            if not sub_id.isdigit() or not all(sup_id.isdigit() for sup_id in sup_ids):
                continue

            sub_id = int(sub_id) - 1
            sub_ids = [
                int(sup_id) - 1 for sup_id in sup_ids if sup_id.strip().isdigit()
            ]

            if sub_ids:
                moss[sub_id, sub_ids] = True

    return moss


def load_rules(rules_file: str) -> Tuple[Dict[str, List], Dict[int, str]]:
    """Load significant rules and SMILES mapping from JSON file."""
    with open(rules_file, "r") as f:
        data = json.load(f)

    rules = data["rules"]
    smiles_map = {int(k): v for k, v in data["smiles_map"].items()}

    return rules, smiles_map


def load_cancer_data(cancer_file: str) -> pd.DataFrame:
    """Load cancer activity labels for molecules."""
    return pd.read_csv(cancer_file, sep=",")


def load_test_molecules(test_csv: str) -> pd.DataFrame:
    """Load test molecule SMILES."""
    return pd.read_csv(test_csv)


def check_substructure_match(mol_smiles: str, substructure_smarts: str) -> bool:
    """Check if a molecule contains a substructure pattern."""
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        pattern = Chem.MolFromSmarts(substructure_smarts)

        if mol is None or pattern is None:
            return False

        return mol.HasSubstructMatch(pattern)
    except Exception:
        return False


def compute_rule_frequencies(
    rules: Dict[str, List], moss_matrix: np.ndarray, C: pd.DataFrame
) -> Dict[Tuple, Dict]:
    """Compute frequencies for goodness measures."""
    n_molecules = moss_matrix.shape[1]
    frequencies = {}

    antecedents = set()
    for rule_size in ["1", "2"]:
        for rule in rules.get(rule_size, []):
            X, cancer_type, MI = rule
            antecedents.add(tuple(sorted(X)))

    for X in antecedents:
        X_mask = np.all(moss_matrix[list(X), :], axis=0)
        fr_X = np.sum(X_mask)

        frequencies[X] = {"fr_X": fr_X, "P_X": fr_X / n_molecules}

        for cancer_type in range(1, 4):
            cancer_column = C[f"anti_cancer_{cancer_type}"].values.astype(bool)
            fr_XC = np.sum(X_mask & cancer_column)
            frequencies[X][f"fr_XC{cancer_type}"] = fr_XC
            frequencies[X][f"P_XC{cancer_type}"] = fr_XC / n_molecules

    frequencies["_global"] = {}
    for cancer_type in range(1, 4):
        cancer_column = C[f"anti_cancer_{cancer_type}"].values.astype(bool)
        fr_C = np.sum(cancer_column)
        frequencies["_global"][f"fr_C{cancer_type}"] = fr_C
        frequencies["_global"][f"P_C{cancer_type}"] = fr_C / n_molecules

    return frequencies


def select_applicable_rules_by_smiles(
    mol_smiles: str,
    rules: Dict[str, List],
    train_smiles_map: Dict[int, str],
) -> Dict[int, List]:
    """
    Select rules where ALL substructures in the rule antecedent
    are present in the test molecule (by checking SMILES patterns).
    """
    applicable_rules = {1: [], 2: [], 3: []}

    for rule_size in ["1", "2"]:
        for rule in rules.get(rule_size, []):
            X, cancer_type, MI = rule

            all_present = True
            for train_sub_id in X:
                substructure_pattern = train_smiles_map.get(train_sub_id)

                if substructure_pattern is None:
                    all_present = False
                    break

                if not check_substructure_match(mol_smiles, substructure_pattern):
                    all_present = False
                    break

            if all_present:
                applicable_rules[cancer_type].append(rule)

    return applicable_rules


def compute_confidence(frequencies: Dict, X: Tuple, cancer_type: int) -> float:
    """Compute confidence: P(C|X) = P(XC) / P(X)"""
    X_tuple = tuple(sorted(X))
    if X_tuple not in frequencies:
        return 0.0

    P_X = frequencies[X_tuple]["P_X"]
    if P_X == 0:
        return 0.0

    P_XC = frequencies[X_tuple].get(f"P_XC{cancer_type}", 0)
    return P_XC / P_X


def compute_lift(frequencies: Dict, X: Tuple, cancer_type: int) -> float:
    """Compute lift: confidence / P(C)"""
    X_tuple = tuple(sorted(X))
    if X_tuple not in frequencies or "_global" not in frequencies:
        return 1.0

    P_XC = frequencies[X_tuple].get(f"P_XC{cancer_type}", 0)
    P_X = frequencies[X_tuple]["P_X"]
    P_C = frequencies["_global"].get(f"P_C{cancer_type}", 0)

    if P_X == 0 or P_C == 0:
        return 1.0

    return P_XC / (P_X * P_C)


def compute_leverage(frequencies: Dict, X: Tuple, cancer_type: int) -> float:
    """Compute leverage: P(XC) - P(X)*P(C)"""
    X_tuple = tuple(sorted(X))
    if X_tuple not in frequencies or "_global" not in frequencies:
        return 0.0

    P_XC = frequencies[X_tuple].get(f"P_XC{cancer_type}", 0)
    P_X = frequencies[X_tuple]["P_X"]
    P_C = frequencies["_global"].get(f"P_C{cancer_type}", 0)

    return P_XC - (P_X * P_C)


def xlogx(x: float) -> float:
    """Helper function for information-theoretic computations."""
    return 0 if x <= 0 else x * np.log2(x)


def compute_mutual_information(frequencies: Dict, X: Tuple, cancer_type: int) -> float:
    """Compute mutual information: MI(X; C)"""
    X_tuple = tuple(sorted(X))
    if X_tuple not in frequencies or "_global" not in frequencies:
        return 0.0

    P_X = frequencies[X_tuple]["P_X"]
    P_XC = frequencies[X_tuple].get(f"P_XC{cancer_type}", 0)
    P_C = frequencies["_global"].get(f"P_C{cancer_type}", 0)

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


def compute_goodness_measures(
    applicable_rules: Dict[int, List],
    frequencies: Dict,
) -> Dict[int, Dict[str, float]]:
    """Compute goodness measures for each cancer type."""
    goodness = {}

    for cancer_type in range(1, 4):
        rules = applicable_rules.get(cancer_type, [])

        if not rules:
            goodness[cancer_type] = {
                "count": 0,
                "avg_confidence": 0.0,
                "avg_MI": 0.0,
                "avg_lift": 1.0,
                "avg_leverage": 0.0,
                "max_confidence": 0.0,
                "max_lift": 1.0,
                "total_MI": 0.0,
            }
            continue

        confidences = []
        mis = []
        lifts = []
        leverages = []

        for rule in rules:
            X, C_type, MI = rule
            X_tuple = tuple(sorted(X))

            conf = compute_confidence(frequencies, X_tuple, cancer_type)
            lift = compute_lift(frequencies, X_tuple, cancer_type)
            leverage = compute_leverage(frequencies, X_tuple, cancer_type)

            confidences.append(conf)
            mis.append(MI)
            lifts.append(lift)
            leverages.append(leverage)

        goodness[cancer_type] = {
            "count": len(rules),
            "avg_confidence": np.mean(confidences),
            "avg_MI": np.mean(mis),
            "avg_lift": np.mean(lifts),
            "avg_leverage": np.mean(leverages),
            "max_confidence": np.max(confidences),
            "max_lift": np.max(lifts),
            "total_MI": np.sum(mis),
        }

    return goodness


def predict_anticancer_activity(
    mol_smiles: str,
    rules: Dict[str, List],
    train_smiles_map: Dict[int, str],
    frequencies: Dict,
    threshold_confidence: float = 0.3,
    threshold_lift: float = 1.0,
) -> Dict:
    """Predict anticancer activities using SMILES-based substructure matching."""
    applicable_rules = select_applicable_rules_by_smiles(
        mol_smiles, rules, train_smiles_map
    )

    goodness = compute_goodness_measures(applicable_rules, frequencies)

    predictions = {}
    for cancer_type in range(1, 4):
        measures = goodness[cancer_type]

        predicted_active = measures["count"] > 0 and (
            measures["avg_confidence"] > threshold_confidence
            or measures["avg_lift"] > threshold_lift
        )

        predictions[cancer_type] = {
            "predicted_active": int(predicted_active),
            "num_applicable_rules": measures["count"],
            "avg_confidence": measures["avg_confidence"],
            "avg_MI": measures["avg_MI"],
            "avg_lift": measures["avg_lift"],
            "avg_leverage": measures["avg_leverage"],
            "max_confidence": measures["max_confidence"],
            "max_lift": measures["max_lift"],
            "total_MI": measures["total_MI"],
        }

    return predictions


def predict_batch(
    moss_train_file: str,
    train_sub_file: str,
    cancer_train_file: str,
    test_molecules_file: str,
    rules_file: str,
    output_file: str,
    threshold_confidence: float = 0.15,
    threshold_lift: float = 0.9,
) -> None:
    """Predict anticancer activities for test molecules."""
    print("Loading training data and rules...")
    moss_train_matrix = get_subgraphs(moss_train_file)
    C_train = load_cancer_data(cancer_train_file)
    rules, train_smiles_map = load_rules(rules_file)

    print("Loading test molecules...")
    test_molecules = load_test_molecules(test_molecules_file)

    print(
        f"Computing frequencies from {moss_train_matrix.shape[1]} training molecules..."
    )
    frequencies = compute_rule_frequencies(rules, moss_train_matrix, C_train)

    print(
        f"Predicting anticancer activities for {len(test_molecules)} test molecules..."
    )
    all_predictions = {}

    for idx, row in test_molecules.iterrows():
        if idx % 10 == 0:
            print(f"  Processing molecule {idx + 1}/{len(test_molecules)}")

        mol_smiles = row["SMILES"]

        predictions = predict_anticancer_activity(
            mol_smiles,
            rules,
            train_smiles_map,
            frequencies,
            threshold_confidence=threshold_confidence,
            threshold_lift=threshold_lift,
        )
        all_predictions[str(idx)] = predictions

    print(f"Saving predictions to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_predictions, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Predict anticancer activities using significant rules"
    )
    parser.add_argument(
        "--moss_train_file",
        type=str,
        default="train.id",
        help="Path to training MOSS output file",
    )
    parser.add_argument(
        "--train_sub_file",
        type=str,
        default="train.sub",
        help="Path to training MOSS subgraph file with SMILES",
    )
    parser.add_argument(
        "--cancer_train_file",
        type=str,
        default="molecule_train.csv",
        help="Path to training cancer data CSV",
    )
    parser.add_argument(
        "--test_molecules_file",
        type=str,
        default="molecule_test.csv",
        help="Path to test molecules CSV with SMILES",
    )
    parser.add_argument(
        "--rules_file",
        type=str,
        default="significant_rules.json",
        help="Path to significant rules JSON",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--threshold_confidence", type=float, default=0.15, help="Confidence threshold"
    )
    parser.add_argument(
        "--threshold_lift", type=float, default=0.9, help="Lift threshold"
    )

    args = parser.parse_args()

    predict_batch(
        args.moss_train_file,
        args.train_sub_file,
        args.cancer_train_file,
        args.test_molecules_file,
        args.rules_file,
        args.output_file,
        threshold_confidence=args.threshold_confidence,
        threshold_lift=args.threshold_lift,
    )
