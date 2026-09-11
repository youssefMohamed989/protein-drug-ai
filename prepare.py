"""Prepare a receptor PDB and ligand SMILES for AutoDock Vina docking.

Receptor: PDBFixer adds missing residues/atoms/hydrogens and removes
heterogens/waters, producing a clean PDB. It's then converted to PDBQT
(the format Vina reads) via Meeko's polymer preparation, which assigns
Gasteiger charges and AutoDock atom types.

Ligand: RDKit embeds a 3D conformer from SMILES (ETKDGv3), optimizes it
with the MMFF94 force field, and Meeko converts it to a flexible-ligand
PDBQT (rotatable bonds auto-detected from the topology).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class PreparedLigand:
    smiles: str
    pdbqt_string: str
    num_rotatable_bonds: int


def prepare_receptor(input_pdb: str, output_pdb: str, ph: float = 7.4) -> str:
    """Clean a raw PDB (add missing atoms/hydrogens, strip waters/heterogens)
    via PDBFixer, and write a receptor-ready PDB. Returns the output path.
    """
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    return output_pdb


def receptor_pdb_to_pdbqt(receptor_pdb: str, output_pdbqt: str, allow_bad_res: bool = True) -> str:
    """Convert a prepared receptor PDB to PDBQT via Meeko's Polymer API.

    IMPORTANT: Meeko perceives bonds from interatomic distances, so it
    expects reasonably relaxed, non-clashing geometry (real crystal
    structures, or anything that has been through at least a short energy
    minimization). Feed it output straight from PDBFixer on a raw/predicted
    structure and it can legitimately fail with "unable to build rdkit mol"
    errors. The standard pipeline in this repo is therefore:

        predict_structure.py -> prepare_receptor() -> run_md.py --minimize_only
        -> receptor_pdb_to_pdbqt()

    i.e. always minimize (src/md/simulation.py) before docking prep. See
    scripts/dock.py, which does this automatically when given a predicted
    (rather than experimental) structure.
    """
    import meeko

    templates = meeko.ResidueChemTemplates.create_from_defaults()
    mk_prep = meeko.MoleculePreparation()
    polymer = meeko.Polymer.from_pdb_file(
        receptor_pdb,
        chem_templates=templates,
        mk_prep=mk_prep,
        allow_bad_res=allow_bad_res,
    )
    pdbqt_string = meeko.PDBQTWriterLegacy.write_string_from_polymer(polymer)
    if isinstance(pdbqt_string, tuple):
        pdbqt_string = pdbqt_string[0]

    Path(output_pdbqt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdbqt, "w") as f:
        f.write(pdbqt_string)
    return output_pdbqt


def prepare_ligand(smiles: str, ph: float = 7.4, seed: int = 42) -> Optional[PreparedLigand]:
    """SMILES -> 3D conformer (RDKit) -> flexible-ligand PDBQT (Meeko)."""
    import meeko
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    embed_result = AllChem.EmbedMolecule(mol, params)
    if embed_result != 0:
        # retry with random coords as a fallback for tricky topologies
        params.useRandomCoords = True
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    mk_prep = meeko.MoleculePreparation()
    mol_setups = mk_prep.prepare(mol)
    result = meeko.PDBQTWriterLegacy.write_string(mol_setups[0])
    # write_string returns (pdbqt_string, success, error_msg)
    pdbqt_string, success, error_msg = result
    if not success:
        raise RuntimeError(f"Meeko ligand preparation failed: {error_msg}")

    n_rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    return PreparedLigand(smiles=smiles, pdbqt_string=pdbqt_string, num_rotatable_bonds=n_rot)


def estimate_binding_box(
    receptor_pdb: str, ligand_resname: Optional[str] = None, padding: float = 8.0
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Estimate a docking search box.

    If `ligand_resname` is given and present as a HETATM in the (raw,
    pre-fixer) receptor PDB, centers the box on that ligand's centroid
    (co-crystallized ligand -> known pocket). Otherwise falls back to the
    geometric center of the whole structure with a generous box — coarser,
    but keeps the pipeline runnable without prior pocket knowledge (a real
    workflow would instead use a pocket-detection tool, e.g. fpocket, or a
    user-supplied center).
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", receptor_pdb)

    coords = []
    ligand_coords = []
    for atom in structure.get_atoms():
        coords.append(atom.get_coord())
        parent_resname = atom.get_parent().get_resname()
        if ligand_resname is not None and parent_resname == ligand_resname:
            ligand_coords.append(atom.get_coord())

    coords = np.array(coords)
    if ligand_coords:
        target = np.array(ligand_coords)
        center = target.mean(axis=0)
        size = (target.max(axis=0) - target.min(axis=0)) + 2 * padding
    else:
        center = coords.mean(axis=0)
        span = coords.max(axis=0) - coords.min(axis=0)
        size = np.minimum(span, np.array([30.0, 30.0, 30.0]))  # cap box size for tractable search

    return tuple(center.tolist()), tuple(size.tolist())
