"""Minimal PDB writer for a reconstructed (N, CA, C, O) backbone trace.

Only backbone atoms are written (no sidechains) — this is the direct
output of `src/structure/nerf.py`. Downstream tools (PDBFixer, in the MD
pipeline) can add missing sidechain atoms and hydrogens from this trace,
matching the standard "predict backbone -> rebuild full-atom -> refine
with MD" pipeline used in this repo (see scripts/predict_structure.py and
scripts/run_md.py).
"""

from __future__ import annotations

from typing import Dict

from src.structure.nerf import BackboneCoordinates

THREE_LETTER: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}


def write_backbone_pdb(backbone: BackboneCoordinates, path: str, chain_id: str = "A") -> None:
    lines = []
    atom_serial = 1
    for i, aa in enumerate(backbone.sequence):
        resname = THREE_LETTER.get(aa.upper(), "UNK")
        res_seq = i + 1
        for atom_name, coords in (
            ("N", backbone.n_coords[i]),
            ("CA", backbone.ca_coords[i]),
            ("C", backbone.c_coords[i]),
            ("O", backbone.o_coords[i]),
        ):
            x, y, z = coords
            element = atom_name[0]
            lines.append(
                f"ATOM  {atom_serial:5d}  {atom_name:<3s}{resname:>4s} "
                f"{chain_id}{res_seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          "
                f"{element:>2s}"
            )
            atom_serial += 1
    lines.append("TER")
    lines.append("END")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
