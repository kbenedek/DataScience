import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def analyze_moss_output():
    """
    Reads the output from the MoSS tool, displays the top 100 frequent
    substructures, and draws the top 5.
    """
    output_filename = "moss_output"

    print(f"--- Reading frequent subgraph data from '{output_filename}' ---")

    try:
        df = pd.read_csv(output_filename, skipinitialspace=True)
    except FileNotFoundError:
        print(f"\nError: '{output_filename}' not found.")
        print(
            "Please make sure you have run the MoSS tool and the output file is in the correct directory."
        )
        return

    df_sorted = df.sort_values(by="s_abs", ascending=False)

    print("\nTop 100 most frequent substructures (sorted by absolute support 's_abs'):")
    with pd.option_context("display.max_rows", 100):
        print(df_sorted.head(100))

    top_5_subgraphs = df_sorted.head(5)

    subgraph_mols = []
    subgraph_labels = []

    print("\n--- Generating 2D images for the 5 most frequent substructures ---")

    for index, row in top_5_subgraphs.iterrows():
        smarts = row["description"]
        mol = Chem.MolFromSmarts(smarts)

        if mol:
            subgraph_mols.append(mol)
            label = f"ID: {row['id']}\nSupport: {row['s_abs']}"
            subgraph_labels.append(label)
        else:
            print(f"Warning: Could not generate molecule from SMARTS pattern: {smarts}")

    if subgraph_mols:
        img = Draw.MolsToGridImage(
            subgraph_mols, molsPerRow=3, subImgSize=(250, 250), legends=subgraph_labels
        )

        image_filename = "moss_top_5_subgraphs.png"
        img.save(image_filename)
        print(f"\nSuccessfully saved the 2D images to '{image_filename}'")
    else:
        print("\nCould not generate any molecules to draw from the top 5 subgraphs.")


if __name__ == "__main__":
    analyze_moss_output()
