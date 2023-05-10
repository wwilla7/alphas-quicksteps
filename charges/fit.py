#!/usr/bin/env python

"""
Fit RESP-style dPol charges to baseline MP2/aug-cc-pvtz ESP data.
"""

import copy
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pint
from rdkit import Chem
from scipy.spatial.distance import cdist

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
import logging

from openff.toolkit import ForceField, Molecule, Topology
from openff.units import unit

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger()


def pair_equivalent(pattern: List) -> np.ndarray:
    """
    A function to pair related patterns together for use as constraints

    Parameters
    ----------
    pattern: List
        A list of patterns, could be elements, SMIRNOFF patterns

    Returns
    -------
    ndarry
        Return pairs of related patterns in a nested numpy ndarry.

    """
    tmp1 = defaultdict(list)
    for idx1, p in enumerate(pattern):
        tmp1[p].append(idx1)

    tmp2 = []
    for key, v in tmp1.items():
        n = len(v)
        if n > 1:
            tmp2.append([[v[i], v[i + 1]] for i in range(n - 1)])
    if len(tmp2) == 0:
        ret = []
    else:
        ret = np.concatenate(tmp2)
    return ret


def coulomb_scaling(rdmol: Chem.rdchem.Mol, coulomb14scale: float = 0.5) -> np.ndarray:
    """

    Parameters
    ----------
    rdmol: Chem.rdchem.Mol
        An input rdkit molecule used for specifying connectivity

    coulomb14scale: float

    Returns
    -------
    ndarray

    """

    natom = rdmol.GetNumAtoms()
    # initializing arrays
    bonds = []
    bound12 = np.zeros((natom, natom))
    bound13 = np.zeros((natom, natom))
    scaling_matrix = np.ones((natom, natom))

    for bond in rdmol.GetBonds():
        b = bond.GetBeginAtomIdx()
        e = bond.GetEndAtomIdx()
        bonds.append([b, e])

    # Find 1-2 scaling_matrix
    for pair in bonds:
        bound12[pair[0], pair[1]] = 12.0
        bound12[pair[1], pair[0]] = 12.0

    # Find 1-3 scaling_matrix
    b13_pairs = []
    for i in range(natom):
        b12_idx = np.nonzero(bound12[i])[0]
        for idx, j in enumerate(b12_idx):
            for k in b12_idx[idx + 1 :]:
                b13_pairs.append([j, k])
    for pair in b13_pairs:
        bound13[pair[0], pair[1]] = 13.0
        bound13[pair[1], pair[0]] = 13.0

    # Find 1-4 scaling_matrix
    b14_pairs = []
    for i in range(natom):
        b12_idx = np.nonzero(bound12[i])[0]
        for j in b12_idx:
            b122_idx = np.nonzero(bound12[j])[0]
            for k in b122_idx:
                for j2 in b12_idx:
                    if k != i and j2 != j:
                        b14_pairs.append([j2, k])

    # Assign coulomb14scaling factor
    for pair in b14_pairs:
        scaling_matrix[pair[0], pair[1]] = coulomb14scale
        scaling_matrix[pair[1], pair[0]] = coulomb14scale

    # Exclude 1-2, 1-3 interactions
    for pair in bonds:
        scaling_matrix[pair[0], pair[1]] = 0.0
        scaling_matrix[pair[1], pair[0]] = 0.0

    for pair in b13_pairs:
        scaling_matrix[pair[0], pair[1]] = 0.0
        scaling_matrix[pair[1], pair[0]] = 0.0

    # Fill 1-1 with zeros
    np.fill_diagonal(scaling_matrix, 0)

    return scaling_matrix


@dataclass
class AMol:
    data_path: str
    off_ff: ForceField
    pol_handler: str = "MPIDPolarizability"

    @property
    def mapped_smiles(self) -> str:
        """
        Returns the mapped SMILES string of the molecule.
        """

        with open(os.path.join(self.data_path, "molecule.smi"), "r") as f:
            smi = f.read()
        return smi

    @property
    def coordinates(self) -> np.ndarray:
        """
        Returns the coordinates of the molecule.
        Unit: bohr
        """

        crds = (
            Q_(
                np.load(os.path.join(self.data_path, "coordinates.npy")),
                ureg.angstrom,
            )
            .to(ureg.bohr)
            .magnitude
        )
        return crds

    @property
    def grid(self) -> np.ndarray:
        """
        Returns the grid of the molecule.
        Unit: bohr
        """
        grid = (
            Q_(np.load(os.path.join(self.data_path, "grid.npy")), ureg.angstrom)
            .to(ureg.bohr)
            .magnitude
        )

        return grid

    @property
    def data_esp(self) -> np.ndarray:
        """
        Returns the ESP data of the molecule.
        Unit: e / bohr
        """
        grid_espi_0 = Q_(
            np.load(os.path.join(self.data_path, "grid_esp.0.npy")),
            ureg.elementary_charge / ureg.bohr,
        ).magnitude

        return grid_espi_0

    @property
    def polarizabilities(self) -> List[str]:
        """
        Returns the polarizabilities of the molecule.
        Unit: a0^3
        """
        mol = Molecule.from_mapped_smiles(self.mapped_smiles)
        parameters = self.off_ff.label_molecules(mol.to_topology())[0]
        ret = [
            Q_(v.epsilon.magnitude, "angstrom**3").to("a0**3").magnitude
            for _, v in parameters[self.pol_handler].items()
        ]
        logger.info(
            f"smiles: {mol.to_smiles(explicit_hydrogens=False)}, converting units to a0^3"
        )
        return ret


def calc_desp(
    mol: AMol,
    qs: np.ndarray,
    alphas: List,
    drjk: np.ndarray,
    r_ij: np.ndarray,
    r_ij3: np.ndarray,
) -> np.ndarray:
    """
    Calculate the ESP from induced dipoles of the molecule.
    Unit: e / bohr
    """

    natom = len(mol.coordinates)

    efield = np.zeros((natom, 3))
    for k in range(natom):
        efield[k] = np.dot(qs, drjk)

    deij = np.einsum("jm, jim->ji", efield, r_ij) * r_ij3.T

    desp = np.dot(alphas, deij)

    return desp


def fit(mol: AMol) -> np.ndarray:
    """
    Fit RESP-style charges to baseline ESPs

    Parameters
    ----------
    mol : AMol

    Returns
    -------
    resp_charges : np.ndarray
    dpol_rrms : float
    npol_rrms : float
    """

    crds = mol.coordinates
    grid = mol.grid
    esps = mol.data_esp.reshape(-1)
    natoms = len(crds)
    npoints = len(grid)

    r_ij = -(grid - crds[:, None])  # distance vector of grid points from atoms
    r_ij0 = cdist(grid, crds, metric="euclidean")
    r_ij1 = np.power(r_ij0, -1)  # euclidean distance of atoms and grids ^ -1
    r_ij3 = np.power(r_ij0, -3)  # euclidean distance of atoms and grids ^ -3

    r_jk = crds - crds[:, None]  # distance vector of atoms from each other
    r_jk1 = cdist(
        crds, crds, metric="euclidean"
    )  # euclidean distance of atoms from each other
    r_jk3 = np.power(
        r_jk1, -3, where=r_jk1 != 0
    )  # euclidean distance of atoms from each other ^ -3

    offmol = Molecule.from_mapped_smiles(mol.mapped_smiles)
    rdmol = offmol.to_rdkit()
    chemically_equivalent_atoms = list(
        Chem.rdmolfiles.CanonicalRankAtoms(rdmol, breakTies=False)
    )
    chemically_equivalent_atoms_pairs = pair_equivalent(chemically_equivalent_atoms)
    n_chemically_equivalent_atoms = len(chemically_equivalent_atoms_pairs)
    net_charge = offmol.total_charge.m_as(unit.elementary_charge)
    coulomb14scale_matrix = coulomb_scaling(rdmol, coulomb14scale=0.5)
    forced_symmetry = set([item for sublist in chemically_equivalent_atoms_pairs for item in sublist])
    polar_region = list(set(range(natoms)) - forced_symmetry)
    alphas = mol.polarizabilities
    elements = [atom.GetSymbol() for atom in rdmol.GetAtoms()]

    for k in range(natoms):
        drjk = r_jk[k] * (r_jk3[k] * coulomb14scale_matrix[k]).reshape(-1, 1)

    ## start charge-fitting
    ## pre-charge, no symmetry
    ndim1 = natoms + 1
    a0 = np.einsum("ij,ik->jk", r_ij1, r_ij1)
    a1 = np.zeros((ndim1, ndim1))
    a1[:natoms, :natoms] = a0

    # Lagrange multiplier
    a1[natoms, :] = 1.0
    a1[:, natoms] = 1.0
    a1[natoms, natoms] = 0.0

    b1 = np.zeros(ndim1)
    b1[:natoms] = np.einsum("ik,i->k", r_ij1, esps)
    b1[natoms] = net_charge

    q1 = np.linalg.solve(a1, b1)[:natoms]

    n_polar_region = len(polar_region)

    q11 = np.zeros(natoms)

    while not np.allclose(q1, q11):
        a10 = copy.deepcopy(a1)
        for j in range(natoms):
            if elements[j] != "H":
                a10[j, j] += 0.0005 * np.power((q1[j] ** 2 + 0.1**2), -0.5)
        q1 = q11
        q11 = np.linalg.solve(a10, b1)[:natoms]

    resp1 = q11

    ndim2 = natoms + 1 + n_chemically_equivalent_atoms + n_polar_region
    a2 = np.zeros((ndim2, ndim2))
    a2[: natoms + 1, : natoms + 1] = a1

    if n_chemically_equivalent_atoms == 0:
        pass
    else:
        for idx, pair in enumerate(chemically_equivalent_atoms_pairs):
            a2[natoms + 1 + idx, pair[0]] = 1.0
            a2[natoms + 1 + idx, pair[1]] = -1.0
            a2[pair[0], natoms + 1 + idx] = 1.0
            a2[pair[1], natoms + 1 + idx] = -1.0

    b2 = np.zeros(ndim2)
    b2[natoms] = net_charge

    charge_to_be_fixed = q1[polar_region]

    for idx, pol_idx in enumerate(polar_region):
        a2[ndim2 - n_polar_region + idx, pol_idx] = 1.0
        a2[pol_idx, ndim2 - n_polar_region + idx] = 1.0
        b2[ndim2 - n_polar_region + idx] = charge_to_be_fixed[idx]

    q2 = resp1
    q22 = np.zeros(natoms)
    while not np.allclose(q2, q22):
        a20 = copy.deepcopy(a2)
        for j in range(natoms):
            if elements[j] != "H":
                a20[j, j] += 0.001 * np.power((q2[j] ** 2 + 0.1**2), -0.5)

        desp = calc_desp(
            mol=mol, qs=q2, alphas=alphas, drjk=drjk, r_ij=r_ij, r_ij3=r_ij3
        )
        esp_to_fit = esps - desp
        b2[:natoms] = np.einsum("ik,i->k", r_ij1, esp_to_fit)
        q2 = q22
        q22 = np.linalg.solve(a20, b2)[:natoms]

    resp2 = q22

    ## quality of fit
    base_esp = np.dot(r_ij1, resp2)
    dpol_esp = calc_desp(
        mol=mol, qs=q2, alphas=alphas, drjk=drjk, r_ij=r_ij, r_ij3=r_ij3
    )
    final_esp = base_esp + dpol_esp
    dpol_rrms = np.sqrt(
        np.sum(np.square(esps - final_esp)) / np.sum(np.square(esps)) / npoints
    )
    npol_rrms = np.sqrt(
        np.sum(np.square(esps - base_esp)) / np.sum(np.square(esps)) / npoints
    )

    return resp2, dpol_rrms, npol_rrms


if __name__ == "__main__":
    from glob import glob

    cwd = os.getcwd()
    data_path = "../data" 
    ff = ForceField("../output.offxml")  # , load_plugins=True)
    pol_handler = "vdW"

    molecules = glob(os.path.join(data_path, "molecule*"))
    nmol = len(molecules)
    for idx, mol in enumerate(molecules):
        logger.info(f"Fitting molecule {idx+1}/{nmol}")
        confs = glob(os.path.join(mol, "conf*"))
        amols = [AMol(conf, ff, pol_handler) for conf in confs]
        for amol in amols:
            resp_qs, dpol_rrms, npol_rrms = fit(amol)
            logger.info(f"dPol RRMS: {dpol_rrms}")
            logger.info(f"nPol RRMS: {npol_rrms}")
