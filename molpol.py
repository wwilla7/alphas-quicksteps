import numpy as np
from openff.toolkit import ForceField, Molecule
from openff.units import unit


def molecular_polarizability(
    smiles: str, off_ff: ForceField, pol_handler: str = "MPIDPolarizability"
):
    offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    symbols = [a.symbol for a in offmol.atoms]

    if np.any([s not in ["C", "H", "O", "N"] for s in symbols]):
        return np.nan
    else:
        pol_parameters = off_ff.label_molecules(offmol.to_topology())[0][pol_handler]
        if pol_handler == "MPIDPolarizability":
            pol_values = [v.polarizability for _, v in pol_parameters.items()]

        elif pol_handler == "vdW":
            pol_values = [
                v.sigma.magnitude * unit.angstrom**3
                for _, v in pol_parameters.items()
            ]

        else:
            raise ValueError(f"Pol handler {pol_handler} not recognized.")

        mol_pol = sum(pol_values)
        return mol_pol.to("angstrom ** 3").magnitude


if __name__ == "__main__":
    import os

    import pandas as pd
    from sklearn.linear_model import LinearRegression

    cwd = os.getcwd()
    data = pd.read_csv("expt_pol.csv", sep=",", skiprows=1)

    ff = ForceField(os.path.join(cwd, "output.offxml"))
    pol_handler = "vdW"
    data["calc."] = data["smiles"].apply(
        lambda x: molecular_polarizability(smiles=x, off_ff=ff, pol_handler=pol_handler)
    )

    plot_data = data.dropna()

    x = plot_data["expt."].values.reshape(-1, 1)
    y = plot_data["calc."].values.reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(x, y)
    r_sq = reg.score(x, y)
    rrms = np.sqrt(np.mean(((y - x) / x) ** 2))
    print(f"Linear regression fit: {reg.coef_[0][0]:.3f}x + {reg.intercept_[0]:.3f}")
    print(f"R^2: {r_sq:.3f}, RRMS: {rrms:.3f}")

    print(f"Saving calculated data to expt_vs_calc_pol.csv")
    plot_data.to_csv("expt_vs_calc_pol.csv", index=True)
