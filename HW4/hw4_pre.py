# Ben's hopes and Dreams
import numpy as np
import pandas as pd


def create_moss_input_file():
    """
    Reads molecule data from molecule.csv and converts it into the
    molecule.db format required by the MoSS tool.

    The output format is: <id>,<value>,<description>
    """
    input_csv = "molecule.csv"
    output_db = "molecule_train.db"

    print(f"--- Reading data from '{input_csv}' ---")
    try:
        # Read the original CSV file
        df = pd.read_csv(input_csv, sep=";")
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return

    print(f"--- Creating '{output_db}' for MoSS ---")

    test_row_idx = np.random.randint(0, len(df))
    print(f"Test row from the input CSV:\n{df.iloc[test_row_idx]}")

    with open(output_db, "w") as f:
        # Iterate over the DataFrame rows and write in the specified format
        for index, row in df.iterrows():
            if index == test_row_idx:
                df.iloc[test_row_idx].to_csv("molecule_test.csv")
                continue
            # <id> will be the row index
            mol_id = index

            # <value> will be from the 'anti_cancer_1' column
            # This value is used by MoSS to split the data.
            value = 0

            # <desc> is the SMILES string
            description = row["SMILES"]

            # Write the formatted line to the file
            f.write(f"{mol_id},{value},{description}\n")

    with open("molecule_test.db", "w") as f:
        row = df.iloc[test_row_idx]
        mol_id = test_row_idx
        value = 0
        description = row["SMILES"]
        f.write(f"{mol_id},{value},{description}\n")
    df.drop(index=test_row_idx).to_csv("molecule_train.csv", index=False)

    print(f"Successfully created '{output_db}' with {len(df)} entries.")
    print("The file is now ready to be used as input for the MoSS tool.")


if __name__ == "__main__":
    create_moss_input_file()
