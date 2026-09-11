"""Reconstruct 3D backbone (N, CA, C) coordinates from per-residue phi/psi
torsion angles via NeRF (Natural Extension Reference Frame), the standard
closed-form method used to place each new atom given the previous three
atoms, a bond length, a bond angle, and a dihedral angle.

Reference: Parsons et al., "Practical conversion from torsion space to
Cartesian space for in silico protein synthesis", J. Comput. Chem. 2005.

This is deliberately independent of any deep-learning framework (pure
NumPy) since it's a deterministic geometric reconstruction step, run once
per prediction rather than as part of the differentiable training graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Idealized backbone bond lengths (Angstrom) and bond angles (degrees),
# standard values from Engh & Huber (1991).
BOND_LEN_N_CA = 1.458
BOND_LEN_CA_C = 1.525
BOND_LEN_C_N = 1.329  # peptide bond to the next residue's N
BOND_LEN_C_O = 1.231

ANGLE_N_CA_C = 111.2  # angle at CA, between N-CA and CA-C
ANGLE_CA_C_N = 116.2  # angle at C, between CA-C and C-N(next)
ANGLE_C_N_CA = 121.7  # angle at N(next), between C(prev)-N and N-CA
ANGLE_CA_C_O = 120.8  # angle at C, between CA-C and C-O (carbonyl)

OMEGA_TRANS = 180.0  # peptide bond dihedral, trans (the overwhelmingly common case)


def _deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0


def place_next_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_length: float,
    bond_angle_deg: float,
    dihedral_deg: float,
) -> np.ndarray:
    """NeRF step: given atoms a-b-c (in placement order), place atom d such
    that bond length c-d, angle b-c-d, and dihedral a-b-c-d match the given
    values.
    """
    bond_angle = _deg2rad(bond_angle_deg)
    dihedral = _deg2rad(dihedral_deg)

    # d2: coordinates of d in the local frame defined by (a, b, c)
    d2 = np.array(
        [
            -bond_length * np.cos(bond_angle),
            bond_length * np.cos(dihedral) * np.sin(bond_angle),
            bond_length * np.sin(dihedral) * np.sin(bond_angle),
        ]
    )

    bc = c - b
    bc_hat = bc / np.linalg.norm(bc)
    ab = b - a
    n = np.cross(ab, bc_hat)
    n_hat = n / np.linalg.norm(n)
    m_hat = np.cross(n_hat, bc_hat)

    M = np.stack([bc_hat, m_hat, n_hat], axis=1)  # 3x3, columns are basis vectors
    d = M @ d2 + c
    return d


@dataclass
class BackboneCoordinates:
    """N/CA/C/O coordinates for each residue, plus the source sequence."""

    sequence: str
    n_coords: np.ndarray  # (L, 3)
    ca_coords: np.ndarray  # (L, 3)
    c_coords: np.ndarray  # (L, 3)
    o_coords: np.ndarray  # (L, 3)


def build_backbone(
    sequence: str,
    phi: Sequence[Optional[float]],
    psi: Sequence[Optional[float]],
    omega: Optional[Sequence[float]] = None,
) -> BackboneCoordinates:
    """Reconstruct a full backbone trace from per-residue torsion angles.

    phi[i] is the N(i)-CA(i)-C(i) rotation using C(i-1); undefined (None)
    for residue 0. psi[i] is undefined for the last residue. Missing values
    are treated as 180 degrees (typical for termini in the absence of
    other information); this only affects the very first/last peptide bond
    placement, not the bulk of the chain.
    """
    L = len(sequence)
    if len(phi) != L or len(psi) != L:
        raise ValueError("phi/psi must have one entry per residue")
    if omega is None:
        omega = [OMEGA_TRANS] * L

    phi = [180.0 if p is None else p for p in phi]
    psi = [180.0 if p is None else p for p in psi]

    n_coords = np.zeros((L, 3))
    ca_coords = np.zeros((L, 3))
    c_coords = np.zeros((L, 3))
    o_coords = np.zeros((L, 3))

    # Seed the first three backbone atoms (N0, CA0, C0) with an arbitrary
    # but geometrically valid placement; everything downstream is built
    # relative to this frame via NeRF.
    n_coords[0] = np.array([0.0, 0.0, 0.0])
    ca_coords[0] = np.array([BOND_LEN_N_CA, 0.0, 0.0])
    # place C0 using the ideal N-CA-C angle in the xy-plane
    theta = _deg2rad(180.0 - ANGLE_N_CA_C)
    c_coords[0] = ca_coords[0] + BOND_LEN_CA_C * np.array([np.cos(theta), np.sin(theta), 0.0])

    for i in range(L):
        if i > 0:
            # N(i) placed from (N(i-1), CA(i-1), C(i-1)) using omega(i-1)
            n_coords[i] = place_next_atom(
                n_coords[i - 1],
                ca_coords[i - 1],
                c_coords[i - 1],
                BOND_LEN_C_N,
                ANGLE_CA_C_N,
                omega[i - 1],
            )
            # CA(i) placed from (CA(i-1), C(i-1), N(i)) using phi(i)
            ca_coords[i] = place_next_atom(
                ca_coords[i - 1],
                c_coords[i - 1],
                n_coords[i],
                BOND_LEN_N_CA,
                ANGLE_C_N_CA,
                phi[i],
            )
            # C(i) placed from (C(i-1), N(i), CA(i)) using psi(i)
            c_coords[i] = place_next_atom(
                c_coords[i - 1],
                n_coords[i],
                ca_coords[i],
                BOND_LEN_CA_C,
                ANGLE_N_CA_C,
                psi[i],
            )
        # Carbonyl O(i): place using (N(i), CA(i), C(i)) with the ideal
        # CA-C-O angle; dihedral N-CA-C-O ~ 180 - psi(i)+180 keeps it
        # trans/planar with the peptide bond for a physically reasonable
        # (if approximate) placement.
        o_dihedral = (psi[i] + 180.0) if i < L - 1 else 180.0
        o_coords[i] = place_next_atom(
            n_coords[i],
            ca_coords[i],
            c_coords[i],
            BOND_LEN_C_O,
            ANGLE_CA_C_O,
            o_dihedral,
        )

    return BackboneCoordinates(
        sequence=sequence,
        n_coords=n_coords,
        ca_coords=ca_coords,
        c_coords=c_coords,
        o_coords=o_coords,
    )


def coordinates_from_pdb_backbone(records: List[dict]) -> "BackboneCoordinates":
    """Helper for round-tripping: build a BackboneCoordinates from a list of
    {'resname':..., 'N':(x,y,z), 'CA':..., 'C':..., 'O':...} dicts, e.g.
    parsed from an existing PDB, for validating reconstruction error.
    """
    seq = "".join(r.get("aa1", "X") for r in records)
    n = np.array([r["N"] for r in records])
    ca = np.array([r["CA"] for r in records])
    c = np.array([r["C"] for r in records])
    o = np.array([r.get("O", r["C"]) for r in records])
    return BackboneCoordinates(sequence=seq, n_coords=n, ca_coords=ca, c_coords=c, o_coords=o)
