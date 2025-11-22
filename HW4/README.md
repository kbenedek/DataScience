# HW4: Anticancer Activity Prediction using Subgraph Mining

This project uses frequent subgraph mining and association rule mining to predict anticancer activity of molecules across three different cancer types.

## Overview

The project consists of several Python scripts that work together to:

1. Prepare molecular data for the MoSS (Molecular Substructure Miner) tool
2. Extract significant association rules from frequent subgraphs
3. Predict anticancer activity for new molecules based on learned rules

## Files

### Main Scripts

- **`hw4_pre.py`**: Preprocessing script that converts molecule data into MoSS-compatible format
  - Reads `molecule.csv` (contains SMILES strings and anticancer labels)
  - Creates `molecule_train.db` and `molecule_test.db` files for MoSS
  - Splits data into training and test sets

- **`hw4_1.py`**: Analysis script for MoSS output
  - Reads MoSS output and displays top 100 frequent substructures
  - Visualizes top 5 substructures using RDKit
  - Generates 2D molecular structure images

- **`rule_extractor.py`**: Extracts significant association rules from subgraph patterns
  - Computes subgraph frequencies and co-occurrences with cancer types
  - Calculates mutual information between subgraphs and anticancer activity
  - Outputs significant rules to `significant_rules.json`

- **`anticancer_predictor.py`**: Predicts anticancer activity for test molecules
  - Loads rules from `significant_rules.json`
  - Uses substructure matching to check if molecules contain rule patterns
  - Generates predictions for three cancer types
  - Outputs results to `predictions.json`

### Data Files

- **`molecule.csv`**: Main dataset with SMILES strings and anticancer activity labels for 3 cancer types
- **`molecule_train.csv`**: Training set molecules
- **`molecule_test.csv`**: Test set molecules
- **`train.id`**: MoSS input file for training data
- **`train.sub`**: MoSS subgraph descriptions (SMILES/SMARTS patterns)
- **`test.id`**: MoSS input file for test data
- **`test.sub`**: MoSS subgraph descriptions for test data

### Output Files

- **`significant_rules.json`**: Extracted association rules with mutual information scores
- **`predictions.json`**: Predicted anticancer activities for test molecules

## Workflow

1. **Data Preparation**:

   ```bash
   python hw4_pre.py
   ```

   Creates MoSS-compatible database files from the molecular data.

2. **Subgraph Mining**:
   Run MoSS tool (external) on the generated `.db` files to identify frequent molecular subgraphs.

3. **Analysis** (optional):

   ```bash
   python hw4_1.py
   ```

   Visualize and analyze the most frequent substructures found by MoSS.

4. **Rule Extraction**:

   ```bash
   python rule_extractor.py --moss_file train.id --sub_file train.sub --cancer_file molecule_train.csv --output significant_rules.json
   ```

   Extract significant association rules between subgraphs and anticancer activity.

5. **Prediction**:

   ```bash
   python anticancer_predictor.py --test_file molecule_test.csv --moss_file test.id --rules_file significant_rules.json --output predictions.json
   ```

   Predict anticancer activity for test molecules.

## Requirements

- Python 3.x
- pandas
- numpy
- RDKit (for molecular structure handling and visualization)
- MoSS tool (external, for frequent subgraph mining)

## Data Format

### Input CSV Format

The `molecule.csv` file contains:

- `anti_cancer_1`: Binary label for cancer type 1
- `anti_cancer_2`: Binary label for cancer type 2
- `anti_cancer_3`: Binary label for cancer type 3
- `SMILES`: SMILES notation of the molecule

### MoSS Format

Files are formatted as: `<id>,<value>,<description>`

- `id`: Molecule identifier
- `value`: Class label (0 or 1)
- `description`: SMILES string

## Key Concepts

- **SMILES**: Simplified Molecular Input Line Entry System - a notation for describing molecular structures
- **SMARTS**: SMILES Arbitrary Target Specification - pattern matching for molecular substructures
- **Subgraph Mining**: Finding frequent structural patterns in molecular graphs
- **Mutual Information**: Measure of association between subgraph presence and anticancer activity
- **Association Rules**: If molecule contains subgraph X, then it has anticancer activity Y
