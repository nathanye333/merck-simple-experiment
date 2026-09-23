"""Dock a molecule against coagulation factor II (F2 / thrombin) using dockstring."""

from dockstring import load_target

# Argatroban — a direct thrombin (F2) inhibitor
SMILES = (
    "C[C@@H]1CCN([C@H](C1)C(=O)O)C(=O)[C@H](CC2=CC=CC=C2)"
    "NS(=O)(=O)C3=CC4=C(C=C3)N=C(N4)N"
)
LIGAND_NAME = "argatroban"

def main() -> None:
    print(f"Loading target F2...")
    target = load_target("F2")

    print(f"Docking {LIGAND_NAME} ({SMILES}) against F2...")
    score, aux = target.dock(SMILES)

    print(f"\nBest docking score (kcal/mol): {score}")
    print(f"All affinities: {aux.get('affinities')}")
    print(f"Best pose (RDKit mol): {aux.get('ligand')}")


if __name__ == "__main__":
    main()
