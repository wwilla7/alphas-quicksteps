import pandas as pd
from openeye import oechem

ifs = oechem.oemolistream("EXPT_POLAR.XYZ")

ifs.SetFormat(oechem.OEFormat_XYZ)

data = [
    {
        "smiles": f"{oechem.OEMolToSmiles(mol)}",
        "expt.(A^3)": mol.GetTitle().split(" ")[1],
    }
    for mol in ifs.GetOEGraphMols()
]


dt = pd.DataFrame(data)

dt.to_csv("expt_pol.csv", index=True, sep=",")
