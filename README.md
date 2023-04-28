# Atomic Polarizability Quick-Step (ALPHAS-QS)
## Fit atomic polarizabilities using SMARTS patterns to reference QM ESP data.

env.yaml: Basic conda environment for running the scripts.

input.offxml: Input file for use to fit polarizabilities. SMARTS patterns are specified in `vdW` handler or `MPIDPolarizability` if applicable.

output.offxml: Output file to store fitted polarizabilities.

[EXPT_POLAR.XYZ](https://github.com/prenlab/amoebaplus_data/blob/master/polarizability/EXPT_POLAR.XYZ): Reference molecular polarizability data curated by AMOEBA+ developers.

oe_process.py: A script to process the above reference data into a format that can be used by SMARTS patterns. 

expt_pol.csv: A CSV file containing the processed reference data.

fit.py: Fit atomic polarizabilities using SMARTS patterns to QM ESP data.

molpol.py A script to calculate molecular polarizabilities from fitted atomic polarizabilities. Including a quick linear regression to reference data with scikit-learn.

data: Data directory for fitting polarizabilities. The included data is for demonstration purposes only. The directory structure is as follows:

```
    data:
        molecule*:
            conf*
                coordinates # unit: Angstrom
                grid # unit: Angstrom
                grid_esp.* # unit: elementary charge/Bohr
                molecule.smi # mapped SMILES string
```
