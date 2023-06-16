#!/usr/bin/env python

import pint
import json
from openff.recharge.charges.bcc import (
    BCCCollection,
    BCCGenerator,
    BCCParameter,
    original_am1bcc_corrections,
)
from openff.recharge.charges.qc import QCChargeGenerator, QCChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.toolkit import Molecule

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

original_bcc_collections = original_am1bcc_corrections()
aromaticity_model = original_bcc_collections.aromaticity_model
parameters = json.load(open("bccs_element.json", "r"))

dpol_collection = BCCCollection(
    parameters=[
        BCCParameter(smirks=sm, value=float(vs)) for sm, vs in parameters.items()
    ]
)

dpol_collection.aromaticity_model = aromaticity_model


def am1bccdpol(offmol: Molecule, bcc_collection=dpol_collection):
    if offmol.conformers:
        conformers = offmol.conformers
    else:
        conformers = ConformerGenerator.generate(
            offmol,
            ConformerSettings(method="omega", sampling_mode="sparse", max_conformers=1),
        )

    am1 = QCChargeGenerator.generate(
        molecule=offmol,
        conformers=conformers,
        settings=QCChargeSettings(theory="am1", sysmmetrize=False, optimize=False),
    )

    assignment_matrix = BCCGenerator.build_assignment_matrix(offmol, bcc_collection)

    pbccs = BCCGenerator.apply_assignment_matrix(
        assignment_matrix=assignment_matrix, bcc_collection=bcc_collection
    )
    ret = am1 + pbccs
    return Q_(ret, "e")
